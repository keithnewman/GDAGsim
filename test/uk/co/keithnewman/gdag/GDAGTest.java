package uk.co.keithnewman.gdag;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

import cern.colt.list.tdouble.DoubleArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tdouble.DoubleFactory1D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.jet.random.tdouble.Normal;
import cern.jet.random.tdouble.engine.DoubleRandomEngine;
import cern.jet.random.tdouble.engine.MersenneTwister64;

public class GDAGTest {
	private GDAG dag;
	private DoubleRandomEngine r = new MersenneTwister64();
	int n = 10000000;

	@Before
	public void setUp() throws Exception {
		// Test prior structure
		dag = GDAG.make(3, r);
		dag.setPermutation((byte) 1);
		dag.addRoot(0, 1, 2);
		dag.addRoot(1, 3, 4);
		Node alpha = Node.make(2);
		alpha.addParent(0, 1);
		alpha.addParent(1, 2);
		dag.addNode(alpha, 2, 0, 1);
	}

	@Test
	public void main() {
		dag.process();

		assertEquals("v0 test failed", 0.5, dag.var(0), 1e-5);
		assertEquals("v1 test failed", 0.25, dag.var(1), 1e-5);
		assertEquals("v2 test failed", 2.5, dag.var(2), 1e-5);

		DoubleMatrix1D s = dag.getMean();
		assertEquals("e0 test failed", 1.0, s.getQuick(0), 1e-5);
		assertEquals("e1 test failed", 3.0, s.getQuick(1), 1e-5);
		assertEquals("e2 test failed", 7.0, s.getQuick(2), 1e-5);

		DoubleMatrix1D total = DoubleFactory1D.dense.make(3);
		DoubleMatrix1D totalNorm = DoubleFactory1D.dense.make(3);
		DoubleMatrix1D Norm = DoubleFactory1D.dense.make(new double[] { 1.0,
				3.0, 7.0 });
		for (int i = 0; i < n; i++) {
			s = dag.nextDouble();
			total.assign(s, cern.jet.math.tdouble.DoubleFunctions.plus);
			s.assign(Norm, cern.jet.math.tdouble.DoubleFunctions.chain(
					cern.jet.math.tdouble.DoubleFunctions.pow(2),
					cern.jet.math.tdouble.DoubleFunctions.minus));
			totalNorm.assign(s, cern.jet.math.tdouble.DoubleFunctions.plus);
		}
		assertEquals("Sample Mean[0] test failed", 1.0, total.getQuick(0)
				/ (double) n, 0.005);
		assertEquals("Sample Mean[1] test failed", 3.0, total.getQuick(1) / n,
				0.005);
		assertEquals("Sample Mean[2] test failed", 7.0, total.getQuick(2) / n,
				0.005);
		assertEquals("Sample Variance[0] test failed", 0.5,
				totalNorm.getQuick(0) / n, 0.01);
		assertEquals("Sample Variance[1] test failed", 0.25,
				totalNorm.getQuick(1) / n, 0.01);
		assertEquals("Sample Variance[2] test failed", 2.5,
				totalNorm.getQuick(2) / n, 0.01);

		// Test with the new built-in simulator in the GDAG class
		total.assign(0);
		totalNorm.assign(0);
		for (int i = 0; i < n; i++) {
			s = dag.nextDouble();
			total.assign(s, cern.jet.math.tdouble.DoubleFunctions.plus);
			s.assign(Norm, cern.jet.math.tdouble.DoubleFunctions.chain(
					cern.jet.math.tdouble.DoubleFunctions.pow(2),
					cern.jet.math.tdouble.DoubleFunctions.minus));
			totalNorm.assign(s, cern.jet.math.tdouble.DoubleFunctions.plus);
		}
		assertEquals("Sample Mean[0] test failed", 1.0, total.getQuick(0)
				/ (double) n, 0.005);
		assertEquals("Sample Mean[1] test failed", 3.0, total.getQuick(1) / n,
				0.005);
		assertEquals("Sample Mean[2] test failed", 7.0, total.getQuick(2) / n,
				0.005);
		assertEquals("Sample Variance[0] test failed", 0.5,
				totalNorm.getQuick(0) / n, 0.01);
		assertEquals("Sample Variance[1] test failed", 0.25,
				totalNorm.getQuick(1) / n, 0.01);
		assertEquals("Sample Variance[2] test failed", 2.5,
				totalNorm.getQuick(2) / n, 0.01);
	}

	@Test
	public void withObservations() {
		dag.priorProcess();
		// Add in observations
		Node alpha = Node.make(2);
		alpha.addParent(1, 1.0);
		alpha.addParent(2, 1.0);
		dag.addObservation(alpha, 5, 0.5, 20);

		dag.process();
		DoubleMatrix1D s = dag.nextDouble();

		Normal gaussianPDF = new Normal(0, Math.sqrt(5.75), r);
		assertEquals("Marginal Log-likelihood failed!",
				Math.log(gaussianPDF.pdf(20-15)), dag.marginalLogLikelihood(),
				1e-5);
		assertEquals("Marginal Log-likelihood failed!",
				Math.log(gaussianPDF.pdf(20-15)), dag.marginalLogLikelihood(),
				1e-5);
		assertEquals("Marginal Log-likelihood failed!",
				Math.log(gaussianPDF.pdf(20-15)), dag.marginalLogLikelihood(),
				1e-5);

		/* cross-check vll against ll, but should really have a proper test! */
		assertEquals("v log-likelihood a  failed!", dag.logLikelihood(),
				dag.vLogLikelihood(s), 1e-5);
		assertEquals("v log-likelihood b failed!", dag.logLikelihood(),
				dag.vLogLikelihood(s), 1e-5);
		assertEquals("v log-likelihood c failed!", dag.logLikelihood(),
				dag.vLogLikelihood(s), 1e-5);

		// Check mean of distribution
		DoubleMatrix1D mean = DoubleFactory1D.dense.make(new double[] {
				33.0 / 23.0, 84.0 / 23.0, 221.0 / 23.0 });
		assertArrayEquals("Mean is not correct!", mean.toArray(), dag.getMean()
				.toArray(), 1e-5);

		assertEquals("v0 test failed", 42.0 / 92.0, dag.var(0), 1e-5);
		assertEquals("v1 test failed", 14.0 / 92.0, dag.var(1), 1e-5);
		assertEquals("v2 test failed", 86.0 / 92.0, dag.var(2), 1e-5);

		DoubleMatrix1D total = DoubleFactory1D.dense.make(3);
		DoubleMatrix1D totalNorm = DoubleFactory1D.dense.make(3);
		for (int i = 0; i < n; i++) {
			s = dag.nextDouble();
			total.assign(s, cern.jet.math.tdouble.DoubleFunctions.plus);
			s.assign(mean, cern.jet.math.tdouble.DoubleFunctions.chain(
					cern.jet.math.tdouble.DoubleFunctions.pow(2),
					cern.jet.math.tdouble.DoubleFunctions.minus));
			totalNorm.assign(s, cern.jet.math.tdouble.DoubleFunctions.plus);
		}
		assertEquals("Sample Mean[0] test failed", 33.0 / 23.0,
				total.getQuick(0) / n, 0.005);
		assertEquals("Sample Mean[1] test failed", 84.0 / 23.0,
				total.getQuick(1) / n, 0.005);
		assertEquals("Sample Mean[2] test failed", 221.0 / 23.0,
				total.getQuick(2) / n, 0.005);
		assertEquals("Sample Variance[0] test failed", 42.0 / 92.0,
				totalNorm.getQuick(0) / n, 0.01);
		assertEquals("Sample Variance[1] test failed", 14.0 / 92.0,
				totalNorm.getQuick(1) / n, 0.01);
		assertEquals("Sample Variance[2] test failed", 86.0 / 92.0,
				totalNorm.getQuick(2) / n, 0.01);

		// Test with the new built-in simulator in the GDAG class
		total.assign(0);
		totalNorm.assign(0);
		for (int i = 0; i < n; i++) {
			s = dag.nextDouble();
			total.assign(s, cern.jet.math.tdouble.DoubleFunctions.plus);
			s.assign(mean, cern.jet.math.tdouble.DoubleFunctions.chain(
					cern.jet.math.tdouble.DoubleFunctions.pow(2),
					cern.jet.math.tdouble.DoubleFunctions.minus));
			totalNorm.assign(s, cern.jet.math.tdouble.DoubleFunctions.plus);
		}
		assertEquals("Sample Mean[0] test failed", 33.0 / 23.0,
				total.getQuick(0) / n, 0.005);
		assertEquals("Sample Mean[1] test failed", 84.0 / 23.0,
				total.getQuick(1) / n, 0.005);
		assertEquals("Sample Mean[2] test failed", 221.0 / 23.0,
				total.getQuick(2) / n, 0.005);
		assertEquals("Sample Variance[0] test failed", 42.0 / 92.0,
				totalNorm.getQuick(0) / n, 0.01);
		assertEquals("Sample Variance[1] test failed", 14.0 / 92.0,
				totalNorm.getQuick(1) / n, 0.01);
		assertEquals("Sample Variance[2] test failed", 86.0 / 92.0,
				totalNorm.getQuick(2) / n, 0.01);
	}

	@Test
	public void spSelfMultOuterTest_DoubleMatrix1D() {
		DoubleMatrix1D sparseVector = DoubleFactory1D.sparse.make(100);
		sparseVector.ensureCapacity(6);
		sparseVector.setQuick(3, -4.7);
		sparseVector.setQuick(14, 3.9);
		sparseVector.setQuick(15, -13.6);
		sparseVector.setQuick(32, -12.1);
		sparseVector.setQuick(64, 60.63);
		sparseVector.setQuick(86, 15.78);
		long time = System.nanoTime();
		DoubleMatrix2D outerProduct = dag.spSelfMultOuter(sparseVector);
		time = System.nanoTime() - time;
		System.out.println("Outer Product computed in " + time * 1.0e-9
				+ " seconds");

		// Extract values
		IntArrayList rows = new IntArrayList(36);
		IntArrayList columns = new IntArrayList(36);
		DoubleArrayList values = new DoubleArrayList(36);
		outerProduct.getNonZeros(rows, columns, values);

		// Compare with actual results.
		int[] targetColumns = new int[] { 3, 14, 15, 32, 64, 86, 3, 14, 15, 32,
				64, 86, 3, 14, 15, 32, 64, 86, 3, 14, 15, 32, 64, 86, 3, 14,
				15, 32, 64, 86, 3, 14, 15, 32, 64, 86 };
		int[] targetRows = new int[] { 3, 3, 3, 3, 3, 3, 14, 14, 14, 14, 14,
				14, 15, 15, 15, 15, 15, 15, 32, 32, 32, 32, 32, 32, 64, 64, 64,
				64, 64, 64, 86, 86, 86, 86, 86, 86 };
		double[] targetValues = new double[] { 22.09, -18.33, 63.92, 56.87,
				-284.961, -74.166, -18.33, 15.21, -53.04, -47.19, 236.457,
				61.542, 63.92, -53.04, 184.96, 164.56, -824.568, -214.608,
				56.87, -47.19, 164.56, 146.41, -733.623, -190.938, -284.961,
				236.457, -824.568, -733.623, 3675.9969, 956.7414, -74.1660,
				61.5420, -214.6080, -190.9380, 956.7414, 249.0084 };
		assertArrayEquals("Outer product has placed values in incorrect row",
				targetRows, rows.elements());
		assertArrayEquals(
				"Outer product has placed values in incorrect column",
				targetColumns, columns.elements());
		assertArrayEquals("Outer product has produced incorrect values",
				targetValues, values.elements(), 1e-6);

		// Check none of the other values are non-zero.
		assertEquals("Incorrect number of non-zero values", 36,
				outerProduct.cardinality());
	}
}
