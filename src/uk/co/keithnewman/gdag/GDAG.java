package uk.co.keithnewman.gdag;

import uk.co.keithnewman.matrixDecompositionAlgo.SparseDoubleCholeskyDecompositionExtended;
import cern.colt.list.tdouble.DoubleArrayList;
import cern.colt.list.tlong.LongArrayList;
import cern.colt.map.tdouble.AbstractLongDoubleMap;
import cern.colt.map.tdouble.OpenLongDoubleHashMap;
import cern.colt.matrix.tdouble.DoubleFactory1D;
import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import cern.jet.random.tdouble.Normal;
import cern.jet.random.tdouble.engine.DoubleRandomEngine;
import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcss;
import edu.emory.mathcs.csparsej.tdouble.Dcs_pinv;

/**
 * Creates the Directed Acyclic Graph for the model.
 * 
 * <p>
 * Status codes are as follows:
 * </p>
 * <ol start="-1">
 * <li>Allocated but not cleared</li>
 * <li>Allocated and cleared</li>
 * <li>Some node definitions, but not full</li>
 * <li>Complete node definitions</li>
 * <li>Processed</li>
 * <li>Sampled</li>
 * </ol>
 * 
 * @author Darren J Wilkinson, Keith Newman
 * @version 0.5.10
 */
public class GDAG {
	/** Value of &frac12;log(2<i>&pi;</i>) */
	private static final double HALF_LOG_2PI = 0.91893853320467274178032973640561763986139747363778;

	private static DoubleRandomEngine rng;

	/** (log|L| - (y-b)L(y-b)) / 2 */
	private double cl = 0;

	/**
	 * Also referred to as <tt>l</tt> in the original C code. Takes the value of
	 * (log|K|) / 2 = sum(log|<i>g<sub>ii</sub></i>|)
	 */
	private double sqrtDetK;

	/**
	 * Also referred to as <tt>ll</tt> in the original C code. Takes value of
	 * log(<i>z<sup>T</sup>z</i>) = sum<sub>i</sub>(z<sub>i</sub><sup>2</sup>)<br>
	 * i.e. dot product of <b>z</b> on itself
	 */
	private double ztz;

	/**
	 * Also referred to as <tt>pl</tt> in the original C code. Takes the value
	 * of (log|K|) / 2 = sum(log|<i>g<sub>ii</sub></i>| -
	 * &frac12;<i>&nu;<sub>i</sub></i><sup>2</sup>) for the <em>prior</em>
	 */
	private double pl;

	/**
	 * Original number of latent variables in the model. Can't exceed this
	 * number if the model is resized during a reset.
	 * 
	 * @since 0.5.01
	 */
	private int nMax;

	/** Current number of latent variables in the model */
	private int n;

	/** Number of latent variables added so far */
	private int count = 0;

	/** The number of observations added so far */
	private int obsCount = 0;

	/**
	 * Whether the precision matrix should be permuted. 0 for no, 1 for yes
	 * 
	 * @since 0.5.02
	 */
	private byte permuteOrder = 0;

	/**
	 * Whether the permutation matrix should be fixed for all future iterations
	 * 
	 * @since 0.5.02
	 */
	private boolean fixedPermutation = false;

	/**
	 * <p>
	 * Current status of the DAG that is being made
	 * </p>
	 * <ol start="-1">
	 * <li>Allocated but not cleared</li>
	 * <li>Allocated and cleared</li>
	 * <li>Some node definitions, but not full</li>
	 * <li>Complete node definitions</li>
	 * <li>Processed</li>
	 * <li>Sampled</li>
	 * </ol>
	 */
	private byte status = -1;

	/**
	 * <p>
	 * First parameter of the canonical representation of the Multivariate
	 * Normal distribution. The parameter is definied as <b>h</b> =
	 * <b>&Sigma;</b><sup>-1</sup><b>&mu;</b>, where <b>&mu;</b> is the mean
	 * parameter of the Multivariate Normal and <b>&Sigma;</b> is the
	 * corresponding covariance matrix.
	 * </p>
	 * <p>
	 * After processing, this turns into the mean vector of a multivariate
	 * Normal distribution (moment parameterisation).
	 * </p>
	 */
	private DoubleMatrix1D h;

	/**
	 * <p>
	 * Temporary vector placeholder. <tt>vv</tt> will stay the same size and
	 * type as <tt>h</tt>, but always treat <tt>vv</tt> values as "contaminated"
	 * at the start of each method you use this placeholder, therefore <b>this
	 * should always be overwritten fully or reset</b> before being used for
	 * calculations.
	 * </p>
	 * <p>
	 * This is being done in preference to the advised practise of minimising
	 * variable scope because in a test, memory usage increased gradually as
	 * instance variables were created repeatedly in a method. Memory usage
	 * remained stable if a placeholder vector was used, with no change in
	 * computational cost. A garbage collection may be enforced on a longer test
	 * which may increased the time cost for clearing memory used by repeated
	 * instantiation of instance variables in methods.
	 * </p>
	 */
	private DoubleMatrix1D vv;

	/**
	 * Second parameter of the canonical representation of the Multivariate
	 * Normal distribution. The parameter is definied as <b>K</b> =
	 * <b>&Sigma;</b><sup>-1</sup>, where <b>&Sigma;</b> is the corresponding
	 * covariance matrix of the Multivariate Normal distribution.
	 */
	private DoubleMatrix2D k;

	/**
	 * Sparse representation of the Cholesky Decomposition of the precision
	 * matrix.
	 */
	private SparseDoubleCholeskyDecompositionExtended chol;

	/** For generating standard Normal samples. */
	private static Normal zGenerator = null;

	/** Symbolic representation of (initial) Cholesky decomposition */
	private Dcss S1;

	/** Inverse permutation for initial Cholesky decomposition */
	private int[] initialPinv;

	/**
	 * Construct a new GDAG model representation
	 * 
	 * @param nLatentVariables
	 *            Number of latent variables to be specified in the model.
	 *            Determines the size of the GDAG object.
	 * @param rng
	 *            Random number generator
	 */
	public GDAG(int nLatentVariables, DoubleRandomEngine r) {
		this.n = nLatentVariables;
		this.nMax = nLatentVariables;
		if (zGenerator == null) {
			rng = r;
			zGenerator = new Normal(0.0, 1.0, rng);
		}
		h = DoubleFactory1D.dense.make(nLatentVariables);
		k = DoubleFactory2D.sparse.make(nLatentVariables, nLatentVariables);
		status = 0;
	}

	/**
	 * Add a new node that has already been defined. N(<i>&alpha;</i><b>x</b> +
	 * <i>b</i>, <i>&tau;</i><sup>-1</sup>)
	 * 
	 * @param alpha
	 *            A sparse vector containing non-zero values in indices
	 *            corresponding to previously defined nodes which are parents to
	 *            this node.
	 * @param i
	 *            The index for this node
	 * @param b
	 *            A constant term added as part of the linear combination.
	 *            <i>b</i> from, <i>&alpha;</i><b>x</b> + <i>b</i>.
	 * @param precision
	 *            Precision value for this node, <i>&tau;</i>.
	 * @throws Error
	 *             Model not cleared or all prior nodes have been specified
	 *             already.
	 */
	protected void addNode(DoubleMatrix1D alpha, int i, double b,
			double precision) throws Error {
		if (status < 0 || status > 1) {
			if (status == 2) {
				System.err.println("Status = 2: All nodes defined."
						+ "You may have attempted to define another "
						+ "node in a full model (this node index "
						+ "would exceed latent dimension)!");
			}
			throw new Error("Error: Status mismatch in addNode.\n");
		}

		// Update the precision matrix
		dusger(precision, alpha);

		int ind;
		LongArrayList keys = ((AbstractLongDoubleMap) alpha.elements()).keys();
		int keySize = keys.size();
		double alpPrec;

		// Update off-diagonal values
		for (int j = 0; j < keySize; j++) {
			ind = (int) keys.getQuick(j);
			alpPrec = alpha.getQuick(ind) * precision;
			k.setQuick(i, ind, k.getQuick(i, ind) - alpPrec);
			k.setQuick(ind, i, k.getQuick(ind, i) - alpPrec);
			h.setQuick(ind, h.getQuick(ind) - b * alpPrec);
		}

		// Update the precision on the diagonal
		k.setQuick(i, i, precision);

		// Update the location vector. Recycle alpPrec to be (b * precision)
		alpPrec = b * precision;
		h.setQuick(i, alpPrec);

		incrementNodeCount();
	}

	/**
	 * Add a new node that has already been defined. N(<i>&alpha;</i><b>x</b> +
	 * <i>b</i>, <i>&tau;</i><sup>-1</sup>)
	 * 
	 * @param Node
	 *            A {@link uk.co.keithnewman.gDAGsim.Node Node} specifying the
	 *            parents of this node.
	 * @param i
	 *            The index for this node
	 * @param b
	 *            A constant term added as part of the linear combination.
	 *            <i>b</i> from, <i>&alpha;</i><b>x</b> + <i>b</i>.
	 * @param precision
	 *            Precision value for this node, <i>&tau;</i>.
	 * @throws Error
	 *             Model not cleared or all prior nodes have been specified
	 *             already.
	 */
	public void addNode(Node node, int i, double b, double precision)
			throws Error {
		addNode(node.getNode(i), i, b, precision);
	}

	/**
	 * Conditions the model on an observation.
	 * 
	 * <i>y<sub>i</sub></i> ~ N(<i>&gamma;</i><b>x</b> + <i>d</i>,
	 * <i>&tau;</i><sup>-1</sup>)
	 * 
	 * @param gamma
	 *            A sparse vector containing non-zero values in indices
	 *            corresponding to previously defined nodes which are parents to
	 *            this node.
	 * @param d
	 *            A constant term added as part of the linear combination.
	 *            <i>d</i> from, <i>&gamma;</i><b>x</b> + <i>d</i>
	 * @param precision
	 *            Precision value for this node, <i>&tau;</i>
	 * @param observation
	 *            Observed value
	 * @throws Error
	 *             No nodes defined or model already processed.
	 */
	protected void addObservation(DoubleMatrix1D gamma, double d,
			double precision, double observation) throws Error {
		if (status < 1 || status > 2) {
			throw new Error("Error: Status mismatch in gdag_add_observation.\n");
		}
		// update precision matrix
		dusger(precision, gamma);

		LongArrayList keys = ((AbstractLongDoubleMap) gamma.elements()).keys();
		int index;
		double obsMinusD = observation - d;
		double dPrecObs = precision * obsMinusD;
		for (int i = 0; i < keys.size(); i++) {
			index = (int) keys.getQuick(i);
			h.setQuick(index, h.getQuick(index) + gamma.getQuick(index)
					* dPrecObs);
		}

		// Update cl (log|L|-(y-b)L(y-b))/2
		cl += 0.5 * (Math.log(precision) - dPrecObs * obsMinusD);

		// Update observation count
		obsCount++;
	}

	/**
	 * Conditions the model on an observation.
	 * 
	 * <i>y<sub>i</sub></i> ~ N(<i>&gamma;</i><b>x</b> + <i>b</i>,
	 * <i>&tau;</i><sup>-1</sup>)
	 * 
	 * @param node
	 *            A {@link uk.co.keithnewman.gDAGsim.Node Node} specifying the
	 *            parents of this node.
	 * @param b
	 *            A constant term added as part of the linear combination.
	 *            <i>b</i> from, <i>&gamma;</i><b>x</b> + <i>b</i>
	 * @param precision
	 *            Precision value for this node, <i>&tau;</i>
	 * @param observation
	 *            Observed value
	 * @throws Error
	 *             No nodes defined or model already processed.
	 */
	public void addObservation(Node node, double b, double precision,
			double observation) throws Error {
		addObservation(node.getNode(n), b, precision, observation);
	}

	/**
	 * Adds a root variable to the model.
	 * 
	 * @param i
	 *            Index of this node.
	 * @param mean
	 *            Mean value of the multivariate Normal mode. Note that the
	 *            "mean parameter" refers to the moment parameterisation of the
	 *            Normal distribution.
	 * @param precision
	 *            Precision value for this node, <i>&tau;</i>
	 * @throws Error
	 *             Model not cleared or all prior nodes have been specified
	 *             already.
	 */
	public void addRoot(int i, double mean, double precision) throws Error {
		if (status < 0 || status > 1) {
			throw new Error("Error: Status mismatch in addRoot.\n");
		}
		h.setQuick(i, precision * mean);
		k.setQuick(i, i, precision);

		incrementNodeCount();
	}

	/**
	 * Creates a vector containing the value 1 in index <code>nodeID</code> and
	 * zero in all other entries.
	 * 
	 * @param nodeID
	 *            Index to contain the only value of 1
	 * @return A vector of zeros, except for a single value of 1 in index
	 *         <code>nodeID</code>.
	 */
	private DoubleMatrix1D basis(int nodeID) {
		// Only used in var, so status > 2 so processed => vv will exist
		vv.assign(0);
		vv.setQuick(nodeID, 1);
		return vv;
	}

	/**
	 * Compute (log|A|)/2, where <b>A</b> is a symmetric positive definite
	 * matrix.
	 * 
	 * @param cholesky
	 *            The Cholesky decomposition of matrix <b>A</b>
	 * @return The determinant of <b>A</b>
	 * @deprecated Since 0.5.04. Use the dedicated method
	 *             {@link uk.co.keithnewman.matrixDecompositionAlgo.SparseDoubleCholeskyDecompositionExtended#getLogDeterminantL()
	 *             getLogDeterminantL()} for this method to prevent <b>L</b>
	 *             being copied.
	 */
	@Deprecated
	private double determinantFromCholesky(
			SparseDoubleCholeskyDecompositionExtended cholesky) {
		return chol.getLogDeterminantL();
	}

	/**
	 * <p>
	 * Plays a part in updating the precision matrix. Calculates the block
	 * matrix,
	 * </p>
	 * <p>
	 * <b>K</b><sub><i>j</i>-1</sub> +
	 * <i>&nu;<sub>j</sub>&alpha;<sub>j</sub>&alpha;
	 * <sub>j</sub><sup>T</sup></i>,
	 * </p>
	 * <p>
	 * according to equation (13) of Wilkinson and Yeung (2004).
	 * </p>
	 * <p>
	 * Automatically updates the current precision matrix with this value.
	 * </p>
	 * 
	 * @param alpha
	 *            The value of <i>&nu;<sub>j</sub></i>
	 * @param x
	 *            The sparse matrix <i>&alpha;<sub>j</sub></i>
	 */
	private void dusger(double nu, DoubleMatrix1D alpha) {
		int thisRow, thisCol;
		LongArrayList keys = ((AbstractLongDoubleMap) alpha.elements()).keys();
		int keySize = keys.size();
		double alp, nuAlphai, nuAlpAlpPlusK;
		for (int i = 0; i < keySize; i++) {
			thisRow = (int) keys.getQuick(i);
			alp = alpha.getQuick(thisRow);
			nuAlphai = nu * alp;
			k.setQuick(thisRow, thisRow, k.getQuick(thisRow, thisRow)
					+ nuAlphai * alp);
			for (int j = i + 1; j < keySize; j++) {
				thisCol = (int) keys.getQuick(j);
				nuAlpAlpPlusK = nuAlphai * alpha.getQuick(thisCol)
						+ k.getQuick(thisRow, thisCol);
				k.setQuick(thisRow, thisCol, nuAlpAlpPlusK);
				k.setQuick(thisCol, thisRow, nuAlpAlpPlusK);
			}
		}
	}

	public double exSq(DoubleMatrix1D alpha) {
		return Math.pow(alpha.zDotProduct(getMean()), 2) + var(alpha);
	}

	/**
	 * Returns the representation of the Cholesky decomposition
	 * 
	 * @return The Cholesky decomposition instance.
	 * @throws Error
	 *             DAG has not been processed.
	 */
	public SparseDoubleCholeskyDecompositionExtended getChol() throws Error {
		if (status < 3) {
			throw new Error(
					"DAG must be processed before returning the Cholesky decomposition.\n");
		}
		return this.chol;
	}

	/**
	 * Return a copy of the <em>lower</em> Cholesky factor of the precision
	 * matrix
	 * 
	 * @return The <em>lower</em> Cholesky factor of the precision matrix
	 */
	public DoubleMatrix2D getCholL() {
		return getCholL(false);
	}

	/**
	 * Return a copy of the Cholesky factor of the precision matrix
	 * 
	 * @param transposeL
	 *            Return the <em>upper</em> Cholesky factor of the precision
	 *            matrix (by transposing the lower Cholesky factor)
	 * @return The Cholesky factor of the precision matrix
	 * @throws Error
	 *             DAG has not been processed.
	 */
	public DoubleMatrix2D getCholL(boolean transposeL) throws Error {
		if (status < 3) {
			throw new Error(
					"DAG must be processed before returning the Cholesky decomposition.\n");
		}
		if (transposeL) {
			return this.chol.getLtranspose();
		} else {
			return this.chol.getL();
		}
	}

	/**
	 * Return the number of latent variables that have been specified in the
	 * GDAG model so far
	 * 
	 * @return Number of latent variables added so far.
	 */
	public int getCount() {
		return this.count;
	}

	/**
	 * Obtain the symbolic analysis of the last Cholesky decomposition
	 * 
	 * @return Symbolic analysis of last Cholesky decomposition performed by
	 *         this instance
	 * @since 0.5.04
	 */
	public Dcss getDcss() {
		return this.chol.getSymbolicAnalysis();
	}

	/**
	 * Don't know if this is safe to use yet.
	 * 
	 * @param alpha
	 * @return
	 */
	@Deprecated
	public double getExSq(DoubleMatrix1D alpha) {
		double result = alpha.zDotProduct(this.h);
		return (result * result + var(alpha));
	}

	/**
	 * Returns the mean of the posterior distribution (moment parameterisation)
	 * 
	 * @return Mean vector (moment parameterisation) of the posterior
	 *         multivariate Normal density.
	 * @throws Error
	 *             DAG has not been processed.
	 */
	public DoubleMatrix1D getMean() {
		if (status < 3) {
			throw new Error(
					"DAG must be processed before returning the mean.\n");
		}
		return this.h;
	}

	/**
	 * Provides the number of observation nodes that have been specified in the
	 * DAG
	 * 
	 * @return The number of observations added
	 */
	public int getObservationCount() {
		return this.obsCount;
	}

	/**
	 * Returns the permutation used for the Cholesky decomposition
	 * 
	 * @return Permutation in use
	 * @since 0.5.03
	 */
	public int[] getP() {
		return Dcs_pinv.cs_pinv(getPinv(), n);
	}

	/**
	 * Returns the inverse permutation used for the Cholesky decomposition
	 * 
	 * @return Inverse permutation in use
	 * @since 0.5.03
	 */
	public int[] getPinv() {
		return chol.getSymbolicAnalysis().pinv;
	}

	/**
	 * Returns the precision matrix of the posterior distribution.
	 * 
	 * @return Precision matrix of the posterior multivariate Normal density.
	 * @throws Error
	 *             DAG has not been processed.
	 */
	public DoubleMatrix2D getPrecision() {
		if (status < 3) {
			throw new Error(
					"DAG must be processed before returning the precision matrix.\n");
		}
		return this.k;
	}

	/**
	 * Returns the size of the GDAG model, as specified on creation of the GDAG
	 * object.
	 * 
	 * @return Dimension of the GDAG model.
	 */
	public int getSize() {
		return this.n;
	}

	/**
	 * Gets the current status of the DAG model
	 * 
	 * @return Current status of the constructed DAG model
	 */
	public byte getStatus() {
		return this.status;
	}

	/**
	 * Tell the model that a new root or node has been defined. This increments
	 * the <tt>count</tt> of how many nodes are currently defined, and updates
	 * the GDAG status to 2 (fully defined) if the model is now full.
	 */
	private void incrementNodeCount() {
		this.count++;
		if (this.count < this.n) {
			this.status = 1;
		} else {
			this.status = 2;
		}
	}

	/**
	 * See Equation (22) of Wilkinson and Yeung (2004).
	 * 
	 * @return The log likelihood of the DAG
	 * @throws Error
	 *             A sample has not yet been taken from the DAG model.
	 */
	public double logLikelihood() throws Error {
		if (status != 4) {
			throw new Error("DAG must be sampled to obtain log-likelihood.\n");
		}
		return (sqrtDetK - 0.5 * ztz - n * HALF_LOG_2PI);
	}

	/**
	 * Factory method to create a new GDAG model representation
	 * 
	 * @param nLatentVariables
	 *            Number of latent variables to be specified in the model.
	 *            Determines the size of the GDAG object.
	 * @param rng
	 *            Random number generator
	 * @return A new GDAG model representation
	 */
	public static GDAG make(int nLatentVariables, DoubleRandomEngine rng) {
		return new GDAG(nLatentVariables, rng);
	}

	/**
	 * See equation (25) of Wilkinson and Yeung (2004)
	 * 
	 * @return The marginal log likelihood of the data,
	 *         log|<i>&pi;(y</i>|<i>&sigma;</i>)|
	 * @throws Error
	 *             DAG has not been processed
	 */
	public double marginalLogLikelihood() throws Error {
		if (status < 3) {
			throw new Error("DAG must be processed to obtain log-likelihood.\n");
		}
		vv = h.copy();
		chol.zMultForDensity(vv);
		double res = vv.zDotProduct(vv);
		res = sqrtDetK - 0.5 * res;
		return (pl + cl - res - obsCount * HALF_LOG_2PI);
	}

	/**
	 * Helper function: Dumps the output of a matrix to <tt>System.out.</tt>
	 * 
	 * @param matrix
	 *            Matrix to dump content of.
	 */
	public void matrixDump(DoubleMatrix2D matrix) {
		System.out.println(matrix.toString());
	}

	/**
	 * Generates a multivariate Normal sample from the posterior distribution of
	 * the latent variables in the GDAG model
	 * 
	 * @return Multivariate Normal sample from the posterior distribution, with
	 *         size equal to the dimension of the GDAG model.
	 * @throws Error
	 *             DAG has not been processed before attempting to generate
	 *             samples.
	 */
	public DoubleMatrix1D nextDouble() throws Error {
		if (status < 3) {
			throw new Error(
					"Status mismatch in nextDouble(): DAG must be processed before samples can be taken.");
		}
		// Fill vv with standard Normal samples.
		for (int i = 0; i < vv.size(); i++) {
			vv.setQuick(i, zGenerator.nextDouble());
		}
		setztz(vv.zDotProduct(vv));
		this.chol.backsolveForSim(vv);
		return vv.assign(this.h, cern.jet.math.tdouble.DoubleFunctions.plus);
	}

	/**
	 * Process the prior structure of the DAG once all prior distributions have
	 * been specified.
	 * 
	 * @throws Error
	 *             Cannot process prior structures when some latent nodes have
	 *             not yet been defined.
	 */
	public void priorProcess() throws Error {
		if (status != 2) {
			throw new Error("DAG is not yet fully defined.\n");
		}
		// This chol and vv will be overwritten when process() is called.
		if (fixedPermutation) {
			try {
				chol = new SparseDoubleCholeskyDecompositionExtended(
						((SparseDoubleMatrix2D) k).getRowCompressed(false), S1);
			} catch (Exception e) { // When initial P-inv doesn't exist, make it
				chol = new SparseDoubleCholeskyDecompositionExtended(
						((SparseDoubleMatrix2D) k).getRowCompressed(false), 1);
				S1 = chol.getSymbolicAnalysis();
				System.out.println("S1 set");
			}
		} else { // no permutation or dynamic permutation
			chol = new SparseDoubleCholeskyDecompositionExtended(
					((SparseDoubleMatrix2D) k).getRowCompressed(false),
					permuteOrder);
		}
		vv = h.copy();
		chol.forwardsolve(vv);

		// compute pl
		pl = chol.getLogDeterminantL() - 0.5 * vv.zDotProduct(vv);
	}

	/**
	 * Process the DAG once all nodes have been specified.
	 * 
	 * @throws Error
	 *             Cannot process prior structures when some latent nodes have
	 *             not yet been defined.
	 * @throws IllegalArgumentException
	 *             if precision matrix is not symmetric positive definite.
	 */
	public void process() throws Error, IllegalArgumentException {
		if (status != 2) {
			throw new Error("DAG is not ready to be processed.\n");
		}
		// Take Cholesky decomposition
		if (fixedPermutation) {
			try {
				chol = new SparseDoubleCholeskyDecompositionExtended(
						((SparseDoubleMatrix2D) k).getRowCompressed(false),
						initialPinv);
			} catch (Exception e) { // When initial P-inv doesn't exist, make it
				chol = new SparseDoubleCholeskyDecompositionExtended(
						((SparseDoubleMatrix2D) k).getRowCompressed(false), 1);
				initialPinv = new int[n];
				System.arraycopy(chol.getSymbolicAnalysis().pinv, 0,
						initialPinv, 0, n);
				System.out.println("pinv set");
			}
		} else { // no permutation or dynamic permutation
			chol = new SparseDoubleCholeskyDecompositionExtended(
					((SparseDoubleMatrix2D) k).getRowCompressed(false),
					permuteOrder);
		}

		chol.solve(h); // This is now the mean, rather than h

		// Compute l (log|K|)/2
		sqrtDetK = chol.getLogDeterminantL();

		// Update status to say GDAG is now processed.
		status = 3;

		// Create a blank placeholder vector vv for future calculations.
		vv = h.like();
	}

	/**
	 * <p>
	 * Set the GDAG to use permutations when performing 'process' methods. This
	 * is used to optimise the Cholesky decomposition.
	 * </p>
	 * <p>
	 * The permutation is specified using a number between 0 and 2:
	 * </p>
	 * <ol start="0">
	 * <li>No permutation</li>
	 * <li>Dynamic permutation (Permutation is recalculated at every iteration)</li>
	 * <li>Fixed permutation (The permutation is calculated once and fixed for
	 * future use)</li>
	 * </ol>
	 * 
	 * @param opt
	 *            Option value of 0, 1 or 2.
	 * @throws IllegalArgumentException
	 *             Invalid permutation option provided, i.e. <tt>opt &lt; 0</tt>
	 *             | <tt>opt &gt; 2</tt>
	 * @since 0.5.02
	 */
	public void setPermutation(byte opt) throws IllegalArgumentException {
		if (opt == 2) { // fixed
			permuteOrder = 1;
			fixedPermutation = true;
		} else if (opt == 1) { // dynamic
			permuteOrder = 1;
			fixedPermutation = false;
		} else if (opt == 0) { // none
			permuteOrder = 0;
			fixedPermutation = false;
		} else {
			throw new IllegalArgumentException(
					"Invalid value provided when setting permutation option.\nChoices are: 0) No permutation; 1) Dynamic permutation; 2) Fixed permutation.\nYou entered \""
							+ opt + "\"");
		}
	}

	/**
	 * Stores the dot-product of the sampled random standard normal values for
	 * density calculations.
	 * 
	 * @param ztz
	 *            Dot product of the standard normal values created when
	 *            generating a sample
	 */
	void setztz(double ztz) {
		this.ztz = ztz;
		status = 4;
	}

	/**
	 * Clears all values to be 0, and resets to status 0. This allows for a GDAG
	 * instance to be recycled providing the new DAG will be the same size
	 * (dimension) as the last.
	 */
	public void reset() {
		status = 0;
		h.assign(0);
		k.assign(0);
		cl = 0;
		sqrtDetK = 0;
		ztz = 0;
		pl = 0;
		count = 0;
		obsCount = 0;
	}

	/**
	 * Clears all values to be 0, resets to status 0 and resizes the model. When
	 * resizing the model, the new size of the latent field cannot exceed the
	 * size of the original model.
	 * 
	 * @since 0.5.01
	 */
	public void reset(int newLatentSize) {
		if (newLatentSize > this.nMax) {
			throw new IllegalArgumentException(
					"Attempt to resize the GDAG to size "
							+ newLatentSize
							+ ", where maximum allowable size is original size of "
							+ nMax + "!");
		}
		status = 0;
		this.n = newLatentSize;
		h = DoubleFactory1D.dense.make(n);
		k = DoubleFactory2D.sparse.make(n, n);
		cl = 0;
		sqrtDetK = 0;
		ztz = 0;
		pl = 0;
		count = 0;
		obsCount = 0;
	}

	/**
	 * Clears all values to be 0, and resets to status 0. This allows for a GDAG
	 * instance to be recycled providing the new DAG will be the same size
	 * (dimension) as the last. Also clears (null) the saved permutation
	 * details.
	 */
	public void resetAll() {
		reset();
		initialPinv = null;
		S1 = null;
	}

	/**
	 * Performs the outer product of a sparse vector on itself. For vector
	 * <b>x</b>, this returns <b>x x</b><sup>T</sup>
	 * 
	 * @param x
	 *            A sparse vector
	 * @return The outer product of <b>x</b> on itself, ie. <b>x
	 *         x</b><sup>T</sup>
	 */
	public DoubleMatrix2D spSelfMultOuter(DoubleMatrix1D x) {
		int rows = (int) x.size(), row, col;
		double rVal, rcVal;
		DoubleMatrix2D A = DoubleFactory2D.sparse.make(rows, rows);
		OpenLongDoubleHashMap map = (OpenLongDoubleHashMap) x.elements();
		LongArrayList keyList = map.keys();
		int xCardinality = (int) keyList.size();
		DoubleArrayList valueList = map.values();
		A.ensureCapacity(xCardinality * xCardinality);
		for (int i = 0; i < xCardinality; i++) {
			row = (int) keyList.getQuick(i);
			rVal = valueList.getQuick(i);
			A.setQuick(row, row, rVal * rVal);
			for (int j = i + 1; j < xCardinality; j++) {
				col = (int) keyList.getQuick(j);
				rcVal = rVal * valueList.getQuick(j);
				A.setQuick(row, col, rcVal);
				A.setQuick(col, row, rcVal);
			}
		}
		return A;
	}

	public double var(DoubleMatrix1D alpha) {
		if (status < 3) {
			throw new Error("Dag must be processed to get var()\n");
		}
		vv.assign(alpha);
		chol.forwardsolve(vv);
		return vv.zDotProduct(vv);
	}

	public double var(int nodeID) {
		return var(basis(nodeID));
	}

	/**
	 * Helper function. Takes differences between the values in the matrix
	 * 
	 * @param vector
	 *            Vector to take difference between elements of.
	 */
	private void vectorDiff(DoubleMatrix1D vector) {
		for (int i = 0; i < vector.size() - 1; i++) {
			vector.setQuick(i, vector.getQuick(i + 1) - vector.getQuick(i));
		}
		vector.setQuick((int) vector.size() - 1, 0.0);
	}

	/**
	 * Helper function: Dumps the output of a vector to <tt>System.out.</tt>
	 * 
	 * @param vector
	 *            Vector to dump the content of.
	 */
	public void vectorDump(DoubleMatrix1D vector) {
		System.out.println(vector.toString());
	}

	/**
	 * Calculates the log-likelihood of the model for a given set of latent
	 * values.
	 * 
	 * @param vector
	 *            Quantiles for use in log-likelihood calculation
	 * @return Log-likelihood at the vector of inputted quantiles.
	 * @throws Error
	 *             Attempting to calculate the log-likelihood before the GDAG
	 *             model has been processed.
	 * @throws IllegalArgumentException
	 *             The inputted vector has a different dimension to the GDAG
	 *             model.
	 */
	public double vLogLikelihood(DoubleMatrix1D vector) {
		if (status < 3) {
			throw new Error("DAG must be processed to obtain vLogLikelihood.\n");
		}
		if (vector.size() != h.size()) {
			throw new IllegalArgumentException(
					"Input vector does not match the size of the DAG.");
		}
		// Need copy or it overwrites input values.
		vv = vector.copy();
		vv.assign(h, cern.jet.math.tdouble.DoubleFunctions.minus);
		chol.zMultForDensity(vv);
		double res = vv.zDotProduct(vv);
		return (sqrtDetK - n * HALF_LOG_2PI - 0.5 * res);
	}
}
