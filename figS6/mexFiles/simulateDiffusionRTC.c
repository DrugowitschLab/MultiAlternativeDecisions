/*
 * simulateDiffusionRTC.c
 *
 * MEX file for simulating diffusion models, using GSL for fast normal RNG.
 *
 * To set the RNG random seed, use
 * 
 * simulateDiffusionRTC(seed)
 * 
 * where seed is a unsigned long int.
 *
 * To draw choices and decision times, use
 *
 * [c, dt] = simulateDiffusionRTC(model_id, theta, Z, cholcovXdt, 
 *                                [intNoise, intFano]);
 *
 * with parameters
 * model_id   - model identifier (integer; see below)
 * theta      - model parameter vector (see below)
 * Z          - trials x N matrix of latent states
 * cholcovXdt - upper triangual cholesky decomposition of dt * covX (LH cov.)
 * intNoise   - gain of integration noise (scalar; optional)
 * intFano    - fano factor of integration noise (scalar; optional)
 *
 * The outputs are
 * c          - 1 x trials vector of choices, 1..N
 * dt         - 1 x trials vector of decision times in seconds
 *
 * The models are
 * 0          - race model & urgency
 *              parameters theta = [u0 b threshold dt maxt]
 *              u0: initial offset (<= threshold)
 *              b : slope of the urgency signal (>= 0)
 * 1          - normalized race model & urgency
 *              parameters theta = [u0 b threshold dt maxt]
 *              b : slope of the urgency signal (>= 0)
 * 2          - contrained race model & urgency
 *              parameters theta = [u0 b a threshold lr iters dt maxt]
 *              u0: initial offset (<= threshold)
 *              b : slope of the urgency signal (>= 0)
 *              a : power of the non-linearity (>= 1)
 *              lr: learning rate for manifold projection (>=0 <=1)
 *              iters: number of iterations for manifold projection (>=1)
 * 3          - as 2, but with nonlinear accumulation
 * 
 * In all models, threshold is the height of the decision threshold, dt is the
 * simulation step size, and maxt is the maximum simulation time. If the
 * threshold is not reached at maxt, then a choice is triggered and dt is set
 * to maxt.
 *
 * Fixed-duration tasks can be simulated by setting threshold = Inf, and maxt
 * to the desired trial duration.
 *
 * Jan Drugowitsch, 2017.
 */

#include "mex.h"
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/* mex argument validation macros */
#define isRealArgument(P) (mxIsDouble(P) &&  !mxIsComplex(P))
#define isRealScalarArgument(P) (isRealArgument(P) && (mxGetNumberOfElements(P) == 1))
#define isReal2DMatrixArgument(P) (isRealArgument(P) && (mxGetNumberOfDimensions(P) == 2))
#define isRealSquareMatrixArgument(P) (isReal2DMatrixArgument(P) && (mxGetM(P) == mxGetN(P)))
#define isRealVectorArgument(P) (isReal2DMatrixArgument(P) && (mxGetM(P) == 1 || mxGetN(P) == 1))

/* constants */
#define INITIAL_NOISE 0.00001

/* RNG state */
static gsl_rng * rng = NULL;

/*
 * RNG management functions
 */

/* function to clean up when mex is cleaned up */
void exitFcn()
{
    if (rng != NULL)
        gsl_rng_free(rng);
}


/* Initializes RNG on first call */
void initialize_rng()
{
    const gsl_rng_type * T;

    if (rng == NULL) {
        gsl_rng_env_setup();
        T = gsl_rng_default;
        rng = gsl_rng_alloc (T);
        /* make sure to clean up rng when mex is cleaned up */
        mexAtExit(exitFcn);
    }
}

/* Sets the RNG seed */
void set_rng_seed(unsigned long int rng_seed)
{
    gsl_rng_set(rng, rng_seed);
}

/* Fills x with correlated random variables.
 *
 * The function assumes x be of size N and U be an upper triagonal
 * matrix of size N x N. It generates z ~ N(0, I) and then returns x = U^T z.
 *
 * It does so without creating a separate z by noting that z[N] is only used to
 * compute x[N], whereas z[N-1] appears only in x[N] and x[N-1], and so on. This
 * means we can use the following update sequence
 * x[N]    = U[N,N] z[N]
 *
 * x[N]   += U[N-1,N] z[N-1]
 * x[N-1]  = U[N-1,N-1] z[N-1]
 *
 * x[N]   += U[N-2,N] z[N-2]
 * x[N-1] += U[N-2,N-1] z[N-2]
 * x[N-2]  = U[N-2,N-2] z[N-2]
 * ...
 * and use z inplace of x without ever overwriting an element of z that is
 * requires at a later stage. 
 */
void ran_corr_x(double* U, int N, double* x)
{
    int i, j;
    double zi;
    /* sample z ~ N(0, I) */
    for (i = 0; i < N; ++i)
        x[i] = gsl_ran_gaussian_ziggurat(rng, 1.0);
    /* compute x = U^T z */
    for (i = N-1; i >= 0; --i) {
        zi = x[i];
        x[i] = U[i + i * N] * zi;
        for (j = i+1; j < N; ++j)
            x[j] += U[i + j * N] * zi;
    }
}


/*
 * helper functions
 **/

/* Returns 1 if either of the N x's crossed the threshold, 0 otherwise. */
int crossed_threshold(double* x, int N, double threshold)
{
    int n;
    for (n = 0; n < N; ++n)
        if (x[n] >= threshold)
            return 1;
    return 0;
}

/* Returns index of last largest of the N x's */
int largest_element(double* x, int N)
{
    int n, nmax;
    nmax = 0;
    for (n = 1; n < N; ++n)
        if (x[n] >= x[nmax])
            nmax = n;
    return nmax;
}

/* Adds constant (returned) such that average of N x's = z */
double normalize_elements(double* x, int N, double z) {
    int i;
    double c;
    c = 0.0;
    for (i = 0; i < N; ++i)
        c += x[i];
    c = z - c/N;    /* c = z - avg(x) */
    for (i = 0; i < N; ++i)
        x[i] += c;
    return c;
}

/* Perform incremental projection onto manifold s.t. ut = mean( f(xn)^alpha )
 *
 * The projection is performed by iterating the below iters times
 * err = ut - mean( f(xn)^alpha )
 * for all n: xn += lr * err
 **/
void project_elements(double* x, int N, double ut, double alpha, double lr, int iters,
    double dtNoise)
{
    int i, n;
    double err, invN;
    invN = 1.0 / N;
    for (i = 0; i < iters; ++i) {
        err = 0.0;
        for (n = 0; n < N; ++n)
            if (x[n] > 0.0)
                err -= pow(x[n], alpha);
        err *= invN;    /* err = lr * (ut - err/N) */
        err += ut;
        err *= lr;
        for (n = 0; n < N; ++n)
            x[n] += err;
        if (dtNoise > 0.0)
            for (n = 0; n < N; ++n)
                x[n] += gsl_ran_gaussian_ziggurat(rng, 
                    sqrt(dtNoise * fabs(x[n]) / iters));
    }
}

/* Same as project_elements(.), but with nonlinearity applied to accumulator
 *
 * After each projection iteration, the accumulators have the non-linearity
 * applied by xn <- f(xn)^alpha.
 */
void project_elements_f(double* x, int N, double ut, double alpha, double lr, int iters,
    double dtNoise)
{
    int i, n;
    double err, invN;
    invN = 1.0 / N;
    for (n = 0; n < N; ++n)
        x[n] = x[n] > 0.0 ? pow(x[n], alpha) : 0.0;
    for (i = 0; i < iters; ++i) {
        err = 0.0;
        for (n = 0; n < N; ++n)
            err -= x[n];   /* don't need f(x[n]), as already applied */
        err *= invN;       /* err = lr * (ut - err/N) */
        err += ut;
        err *= lr;
        for (n = 0; n < N; ++n)   /* x[n] <- f(x[n] + err) */
            x[n] = x[n] > -err ? pow(x[n] + err, alpha) : 0.0;
        if (dtNoise > 0.0)
            for (n = 0; n < N; ++n)
                x[n] += gsl_ran_gaussian_ziggurat(rng, 
                    sqrt(dtNoise * fabs(x[n]) / iters));        
    }
}

/* Perform projection onto manifold s.t. ut = mean( f(xn)^alpha)
 *
 * In contrast to project_elements(.) that uses an iteration, this function
 * uses a Taylor expansion to approximate the projection.
 **/
void project_elements_approx(double* x, int N, double ut, double alpha)
{
    int n;
    double s1, s2, s3;
    s1 = 0.0;
    for (n = 0; n < N; ++n)
        if (x[n] > 0.0)
            s1 += pow(x[n], alpha);
    if (s1 == 0.0)
        s1 = pow(ut, 1/alpha);     /* sum x[n] = 0 -> use shortcut */
    else {
        s2 = 0.0;                  /* s2 = sum x[n]^(alpha - 2) */
        s3 = 0.0;                  /* s3 = sum x[n]^(alpha - 3) */
        for (n = 0; n < N; ++n)
            if (x[n] > 0.0) {
                s2 += pow(x[n], alpha-1);
                s3 += pow(x[n], alpha-2);
            }
        s1 = (sqrt(s2 * s2 - 2 * (alpha - 1) / alpha * s3 * (s1 - N * ut))
             - s2) / ((alpha - 1) * s3);
    }
    for (n = 0; n < N; ++n)
        x[n] += s1;
}

/* Adds new, noisy evidence to the N x's.
 *
 * It adds dz[n] to each x[n], and in addition correlated noise with covariance
 * specified by cholcovXdt. If dtNoise > 0.0, it additionally adds integration
 * noise with variance dtNoise * x[n]^2. dx is a temporary vector for the noise
 * of size N.
 */
void add_evidence(double* x, double* dz, double* cholcovXdt, int N,
                  double dtNoise, double* dx)
{
    int n;
    ran_corr_x(cholcovXdt, N, dx);
    for (n = 0; n < N; ++n)
        if (dtNoise > 0.0) {
            x[n] += dz[n] + dx[n] + gsl_ran_gaussian_ziggurat(rng, 
                sqrt(dtNoise * fabs(x[n])));
        } else {
            x[n] += dz[n] + dx[n];
        }
}


/*
 * race model & urgency
 */
void simURM(double u0, double b, double threshold, double dt, double maxt,
            double* Z, double* cholcovXdt, int N, int trials,
            double intNoise, double intFano, double* outC, double* outDt)
{
    int trial, n, i, maxi;
    double dtNoise;
    double *x, *dx, *dz;

    maxi = (int) ceil(maxt / dt);
    dtNoise = dt * intFano * intNoise * intNoise;

    x = malloc(N * sizeof(double));
    dx = malloc(N * sizeof(double));
    dz = malloc(N * sizeof(double));
    if (x == NULL || dx == NULL || dz == NULL) {
        free(x);
        free(dx);
        free(dz);
        mexErrMsgIdAndTxt("simulateDiffusionRTC:outOfMemory",
            "Failed to allocate memory");
    }

    for (trial = 0; trial < trials; ++trial) {
        for (n = 0; n < N; ++n) {
            dz[n] = dt * Z[trial + trials*n] + b * dt;
            x[n] = u0 + gsl_ran_gaussian_ziggurat(rng, INITIAL_NOISE);
        }
        for (i = 1; i <= maxi; ++i) {
            if (crossed_threshold(x, N, threshold)) {
                outDt[trial] = (i-1)*dt;
                goto crossed;
            }
            add_evidence(x, dz, cholcovXdt, N, dtNoise, dx);
        }
        /* threshold not crossed before last iteration */
        outDt[trial] = maxt;
        crossed:
        outC[trial] = largest_element(x, N) + 1;
    } /* trial */

    free(x);
    free(dx);
    free(dz);
}


/*
 * normalized race model & urgency
 */
void simUNRM(double u0, double b, double threshold, double dt, double maxt,
             double* Z, double* cholcovXdt, int N, int trials,
             double intNoise, double intFano, double* outC, double* outDt)
{
    int trial, n, i, maxi;
    double dtNoise;
    double *x, *dx, *dz;

    maxi = (int) ceil(maxt / dt);
    dtNoise = dt * intFano * intNoise * intNoise;

    x = malloc(N * sizeof(double));
    dx = malloc(N * sizeof(double));
    dz = malloc(N * sizeof(double));
    if (x == NULL || dx == NULL || dz == NULL) {
        free(x);
        free(dx);
        free(dz);
        mexErrMsgIdAndTxt("simulateDiffusionRTC:outOfMemory",
            "Failed to allocate memory");
    }

    for (trial = 0; trial < trials; ++trial) {
        for (n = 0; n < N; ++n) {
            dz[n] = dt * Z[trial + trials*n];
            x[n] = gsl_ran_gaussian_ziggurat(rng, INITIAL_NOISE);
        }
        normalize_elements(x, N, u0);
        for (i = 1; i <= maxi; ++i) {
            if (crossed_threshold(x, N, threshold)) {
                outDt[trial] = (i-1)*dt;
                goto crossed;
            }
            add_evidence(x, dz, cholcovXdt, N, dtNoise, dx);
            normalize_elements(x, N, u0 + b*dt*i);
        }
        /* threshold not crossed before last iteration */
        outDt[trial] = maxt;
        crossed:
        outC[trial] = largest_element(x, N) + 1;
    } /* trial */

    free(x);
    free(dx);
    free(dz);
}


/*
 * race model without threshold
 */
void simNoThresh(double dt, double maxt, double* Z, double* cholcovXdt,
                 int N, int trials, double* outC, double* outDt)
{
    int trial, n;
    double dtscale;
    double *x;

    dtscale = sqrt(maxt / dt);

    x = malloc(N * sizeof(double));
    if (x == NULL) {
        mexErrMsgIdAndTxt("simulateDiffusionRTC:outOfMemory",
            "Failed to allocate memory");
    }

    for (trial = 0; trial < trials; ++trial) {
        /* draw single correlated sample and rescale from dt to maxt */
        ran_corr_x(cholcovXdt, N, x);
        for (n = 0; n < N; ++n) {
            x[n] = x[n] * dtscale + Z[trial + trials*n] * maxt;
        }
        outDt[trial] = maxt;
        outC[trial] = largest_element(x, N) + 1;
    }

    free(x);
}


/*
 * constrained race model & urgency
 */
void simUCRM(double u0, double b, double a, double threshold, double lr, int iters,
             double dt, double maxt, double* Z, double* cholcovXdt, int N, int trials,
             double intNoise, double intFano, double* outC, double* outDt)
{
    int trial, n, i, maxi;
    double dtNoise;
    double *x, *dx, *dz;

    /* accumulation is constrained to u(t) = mean_n | f(x_n) |^a. The RHS is at
     * most f(x_n)^a / N for the largest x_n, such that the threshold can only
     * every be reached if max_t u(t) >= threshold^a / N. For higher thresholds,
     * we can ignore the contraint and simply accumulate with a much simpler,
     * unbounded procedure. This only works without integration noise, as the
     * latter depends on the accumulator state, which is impacted by the
     * constraint.
     */
    if ((intNoise <= 0.0) && (N * (u0 + b * maxt) < pow(threshold, a))) {
        simNoThresh(dt, maxt, Z, cholcovXdt, N, trials, outC, outDt);
        return;
    }

    maxi = (int) ceil(maxt / dt);
    dtNoise = dt * intFano * intNoise * intNoise;

    x = malloc(N * sizeof(double));
    dx = malloc(N * sizeof(double));
    dz = malloc(N * sizeof(double));
    if (x == NULL || dx == NULL || dz == NULL) {
        free(x);
        free(dx);
        free(dz);
        mexErrMsgIdAndTxt("simulateDiffusionRTC:outOfMemory",
            "Failed to allocate memory");
    }

    for (trial = 0; trial < trials; ++trial) {
        for (n = 0; n < N; ++n) {
            dz[n] = dt * Z[trial + trials*n];
            x[n] = u0 + gsl_ran_gaussian_ziggurat(rng, INITIAL_NOISE);
        }
        project_elements(x, N, u0, a, lr, iters, dtNoise);
/*        project_elements_approx(x, N, u0, a); */
        for (i = 1; i <= maxi; ++i) {
            if (crossed_threshold(x, N, threshold)) {
                outDt[trial] = (i-1)*dt;
                goto crossed;
            }
            add_evidence(x, dz, cholcovXdt, N, 0.0, dx);
            project_elements(x, N, u0 + b*dt*i, a, lr, iters, dtNoise);
/*            project_elements_approx(x, N, u0 + b*dt*i, a); */
        }
        /* threshold not crossed before last iteration */
        outDt[trial] = maxt;
        crossed:
        outC[trial] = largest_element(x, N) + 1;
    } /* trial */

    free(x);
    free(dx);
    free(dz);
}


/*
 * constrained race model & urgency, with nonlinear accumulation
 */
void simUCRM2(double u0, double b, double a, double threshold, double lr, int iters,
              double dt, double maxt, double* Z, double* cholcovXdt, int N, int trials,
              double intNoise, double intFano, double* outC, double* outDt)
{
    int trial, n, i, maxi;
    double dtNoise;
    double *x, *dx, *dz;

    /* unlike in simUCRM(.), we cannot shortcut simulations if boundaries
     * might not be hit, as the accumulation itself is non-linear and cannot
     * simply be summarised.
     */
    maxi = (int) ceil(maxt / dt);
    dtNoise = dt * intFano * intNoise * intNoise;

    x = malloc(N * sizeof(double));
    dx = malloc(N * sizeof(double));
    dz = malloc(N * sizeof(double));
    if (x == NULL || dx == NULL || dz == NULL) {
        free(x);
        free(dx);
        free(dz);
        mexErrMsgIdAndTxt("simulateDiffusionRTC:outOfMemory",
            "Failed to allocate memory");
    }

    for (trial = 0; trial < trials; ++trial) {
        for (n = 0; n < N; ++n) {
            dz[n] = dt * Z[trial + trials*n];
            x[n] = u0 + gsl_ran_gaussian_ziggurat(rng, INITIAL_NOISE);
        }
        project_elements_f(x, N, u0, a, lr, iters, dtNoise);
        for (i = 1; i <= maxi; ++i) {
            if (crossed_threshold(x, N, threshold)) {
                outDt[trial] = (i-1)*dt;
                goto crossed;
            }
            add_evidence(x, dz, cholcovXdt, N, 0.0, dx);
            project_elements_f(x, N, u0 + b*dt*i, a, lr, iters, dtNoise);
        }
        /* threshold not crossed before last iteration */
        outDt[trial] = maxt;
        crossed:
        outC[trial] = largest_element(x, N) + 1;
    } /* trial */

    free(x);
    free(dx);
    free(dz);
}


/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    initialize_rng();
    int model_id, N, trials, nTheta;
    double *theta, *Z, *cholcovXdt, *outC, *outDt;
    double intNoise, intFano;

    if (nrhs == 1) {
        /* set RNG seed */
        if (!isRealScalarArgument(prhs[0]))
            mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                "Single real scalar input expected");
        set_rng_seed((unsigned long int) mxGetScalar(prhs[0]));

    } else if (nrhs == 4 || nrhs == 6) {
        /* validate arguments */
        if (!isRealScalarArgument(prhs[0]))
            mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                "First argument expected to be real scalar");
        if (!isRealVectorArgument(prhs[1]))
            mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                "Second argument expected to be real vector");
        if (!isReal2DMatrixArgument(prhs[2]))
            mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                "Third argument expected to be real matrix");
        if (!isRealSquareMatrixArgument(prhs[3]))
            mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                "Fourth argument expected to be real square matrix");
        if (nrhs == 6) {
            if (!isRealScalarArgument(prhs[4]))
                mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                    "Fifth argument expected to be real scalar");
            if (!isRealScalarArgument(prhs[5]))
                mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                    "Sixth argument expected to be real scalar");
        }

        /* model-independent settings */
        nTheta = mxGetNumberOfElements(prhs[1]);
        if (nTheta < 4)
            mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                "Second argument expected to have at least four elements");
        theta = mxGetPr(prhs[1]);
        if (theta[nTheta-2] <= 0.0)
            mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                "dt > 0 expected");            
        if (theta[nTheta-1] <= theta[nTheta-2])
            mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                "maxt > dt expected");
        trials = mxGetM(prhs[2]);
        N = mxGetN(prhs[2]);
        if (N < 2)
            mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                "Third argument expected to have at least two columns");
        if (trials < 1)
            mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                "Third argument expected to have at least two columns");
        if (mxGetN(prhs[3]) != N)
            mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                "Fourth argument expected to be of size N x N");
        Z = mxGetPr(prhs[2]);
        cholcovXdt = mxGetPr(prhs[3]);
        if (nrhs == 6) {
            intNoise = mxGetScalar(prhs[4]);
            intFano = mxGetScalar(prhs[5]);
            if (intNoise < 0.0)
                mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                    "Fifth argument needs to be non-negative");
            if (intFano < 0.0)
                mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                    "Sixth argument needs to be non-negative");                
        } else {
            intNoise = 0.0;
            intFano = 0.0;
        }

        /* check and allocate output */
        if (nlhs != 2)
            mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongOutputs",
                "Two outputs expected");            
        plhs[0] = mxCreateDoubleMatrix(1, trials, mxREAL);
        outC = mxGetPr(plhs[0]);
        plhs[1] = mxCreateDoubleMatrix(1, trials, mxREAL);
        outDt = mxGetPr(plhs[1]);

        /* model-dependent settings */
        model_id = (int) mxGetScalar(prhs[0]);
        if (model_id < 0 || model_id > 3)
            mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                "First argument needs to be between 0 and 3");
        switch (model_id) {
            case 0 :
                /* race model & urgency, theta = [u0 b threshold dt maxt] */
                if (nTheta != 5)
                    mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                        "Expected parameter vector with five elements");
                if (theta[0] > theta[2])
                    mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                        "u0 <= threshold expected");
                if (theta[1] < 0.0)
                    mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                        "b >= 0 expected");
                simURM(theta[0], theta[1], theta[2], theta[3], theta[4],
                       Z, cholcovXdt, N, trials, intNoise, intFano, outC, outDt);
                break;

            case 1 :
                /* race model & normalization, theta = [u0 b threshold dt maxt] */
                if (nTheta != 5)
                    mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                        "Expected parameter vector with four elements");
                if (theta[0] > theta[2])
                    mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                        "u0 <= threshold expected");
                if (theta[1] < 0.0)
                    mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                        "b >= 0 expected");
                simUNRM(theta[0], theta[1], theta[2], theta[3], theta[4],
                        Z, cholcovXdt, N, trials, intNoise, intFano, outC, outDt);
                break;

            case 2 :
                /* constrained race model & urgency, theta = [u0 b a threshold lr iters dt maxt] */
                if (nTheta != 8)
                    mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                        "Expected parameter vector with eight elements");
                if (theta[0] > theta[3])
                    mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                        "u0 <= threshold expected");
                if (theta[1] < 0.0)
                    mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                        "b >= 0 expected");
                if (theta[2] < 1.0)
                    mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                        "a >= 1 expected");
                if (theta[4] < 0.0 || theta[4] > 1.0)
                    mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                        "0 <= lr <= 1 expected");
                if (theta[5] < 1.0)
                    mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                        "iters >= 1 expected");
                simUCRM(theta[0], theta[1], theta[2], theta[3], theta[4],
                    (int) theta[5], theta[6], theta[7], Z, cholcovXdt, N, trials,
                    intNoise, intFano, outC, outDt);
                break;

            case 3 :
                /* constrained race model & urgency, theta = [u0 b a threshold lr iters dt maxt] */
                if (nTheta != 8)
                    mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                        "Expected parameter vector with eight elements");
                if (theta[0] > theta[3])
                    mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                        "u0 <= threshold expected");
                if (theta[1] < 0.0)
                    mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                        "b >= 0 expected");
                if (theta[2] < 1.0)
                    mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                        "a >= 1 expected");
                if (theta[4] < 0.0 || theta[4] > 1.0)
                    mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                        "0 <= lr <= 1 expected");
                if (theta[5] < 1.0)
                    mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                        "iters >= 1 expected");
                simUCRM2(theta[0], theta[1], theta[2], theta[3], theta[4],
                    (int) theta[5], theta[6], theta[7], Z, cholcovXdt, N, trials,
                    intNoise, intFano, outC, outDt);
                break;

            default :
                mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
                    "First argument needs to be between 0 and 3");

        }
    } else {
        mexErrMsgIdAndTxt("simulateDiffusionRTC:wrongInputs",
            "Either one, four, or six inputs required.");
    }
}
