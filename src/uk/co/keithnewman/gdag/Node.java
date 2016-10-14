package uk.co.keithnewman.gdag;

import cern.colt.matrix.tdouble.DoubleFactory1D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tint.IntFactory1D;
import cern.colt.matrix.tint.IntMatrix1D;

/**
 * <p>
 * Defines the parents for a node that will be added to the DAG. When all the
 * parents have been defined, these details will be exported to the DAG
 * specification.
 * </p>
 * 
 * <p>
 * Note that observations are also added to the DAG as a node. When adding an
 * observation to the model, specify the parents using a node specification from
 * this class.
 * </p>
 * 
 * @author Keith Newman, Darren J. Wilkinson
 * 
 */
public class Node {
	/**
	 * Sparse vector to contain the egde strengths defining the relationships
	 * between a node and its parent(s)
	 */
	private DoubleMatrix1D a;

	/**
	 * Dense vector storing the indices of this node's parents, corresponding to
	 * each of the edge strengths given in
	 * {@link uk.co.keithnewman.gDAGsim.Node.a a}
	 */
	private IntMatrix1D index;

	/**
	 * Counter to track how many of the parents for this node have been
	 * specified so far.
	 */
	private int i;

	/**
	 * Allocates memory for a new node to be defined.
	 * 
	 * @param numberOfParents
	 *            Number of parents for the new node about to be defined.
	 */
	public Node(int numberOfParents) {
		reset(numberOfParents);
	}

	/**
	 * Define the parent of the node being constructed.
	 * 
	 * @param parentID
	 *            The ID number for the node being created.
	 * @param alpha
	 *            The coefficient for the edge defining the relationship between
	 *            this node and its parent.
	 * @throws ArrayIndexOutOfBoundsException
	 *             Attempt to add more parents than the size of this node.
	 */
	public void addParent(int parentID, double alpha)
			throws ArrayIndexOutOfBoundsException {
		try {
			index.setQuick(i, parentID);
			a.setQuick(i, alpha);
			i++;
		} catch (ArrayIndexOutOfBoundsException e) {
			System.err
					.println("Too many parents have been added to this node.  This node can have "
							+ a.size()
							+ " parents but "
							+ (i + 1)
							+ " parents were specified before this error was found.\nUse method clear() if a previous node specification needs to be deleted before specifying the next node.\nUse method reset(int numberOfParents) to clear a previously created node and resize to a new size before specifying the next node.");
			throw e;
		}
	}

	/**
	 * Exports the current node as a sparse matrix.
	 * 
	 * @param sizeOfDAG
	 *            Total number of nodes in the DAG.
	 * @return The sparse vector defining the parents of this Node. This is
	 *         defined as <i>&alpha;<sub>j</sub></i> in Equation (13) of
	 *         Wilkinson and Yeung (2004)
	 * @throws IllegalArgumentException
	 *             More parent nodes have been specified than the number of
	 *             variables in the DAG.
	 * @throws Error
	 *             Attempting to retrieve a node when some parents have not yet
	 *             been defined.
	 */
	public DoubleMatrix1D getNode(int sizeOfDAG)
			throws IllegalArgumentException, Error {
		if (i > sizeOfDAG) {
			throw new IllegalArgumentException(
					"The number of parents defined here is greater than the number of nodes in the DAG");
		}
		// Check Node has been fully defined.
		if (i != a.size()) {
			throw new Error(
					"Attempting to define a node where all parents have not been defined.\nExpected "
							+ a.size()
							+ " parents, found "
							+ i
							+ " defined parents.");
		}
		DoubleMatrix1D node = DoubleFactory1D.sparse.make(sizeOfDAG);
		node.ensureCapacity(i);
		for (int j = 0; j < i; j++) {
			node.setQuick(index.getQuick(j), a.getQuick(j));
		}
		return node;
	}

	/**
	 * Clears the values and prepares for a new node definition. Use this when
	 * the current node has been added to the DAG and you no longer need it.
	 * 
	 * @param numberOfParents
	 *            Number of parents for the new node about to be defined.
	 */
	public void reset(int numberOfParents) {
		a = DoubleFactory1D.dense.make(numberOfParents);
		index = IntFactory1D.dense.make(numberOfParents);
		i = 0;
	}

	/**
	 * Clears the values and prepares for a new node definition <i>with the same
	 * number of parents</i>. Use this when the current node has been added to
	 * the DAG and you no longer need it.
	 */
	public void clear() {
		a.assign(0);
		index.assign(0);
		i = 0;
	}

	/**
	 * Factory method to allocate memory for a new Node to be defined.
	 * 
	 * @param numberOfParents
	 *            Number of parents for the new node about to be defined.
	 * @return A node that is now ready to be defined.
	 */
	public static Node make(int numberOfParents) {
		return new Node(numberOfParents);
	}
}
