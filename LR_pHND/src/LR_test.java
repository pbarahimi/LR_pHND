import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import Jama.Matrix;
import gurobi.*;

public class LR_test {
	private static GRBModel model;
	private static Matrix C;
	private static HashMap<String, Integer> varHash = new HashMap<String, Integer>();
	private static int k = 0;
	private static double miu = 0.5;
	private static double epsilon = 1;
	private static double UB = 14;
	
	
	/**
	 * prints the solution
	 * @param model
	 * @throws GRBException
	 */
	private static void printSol() throws GRBException{
		for (GRBVar var : model.getVars()) System.out.println(var.get(GRB.StringAttr.VarName) + " : " + var.get(GRB.DoubleAttr.X));
	}
	
	/**
	 * Updates the objective function coefficients based on the Lagrangian Multipliers put in.
	 * @param model
	 * @param uHash
	 * @param U
	 * @param C
	 * @param d
	 * @param D
	 * @throws GRBException
	 */
	private static void updateObjCoeffs(List<Matrix> U, List<Matrix> d, List<Matrix> D) throws GRBException{
		double[][] temp =  C.getArrayCopy();
		Matrix updatedC = new Matrix(temp);
		double constant = 0;
		for (int i = 0 ; i<U.size() ; i++){
			updatedC = updatedC.minus(U.get(i).transpose().times(D.get(i)));			
			constant += (U.get(i).transpose().times(d.get(i))).get(0, 0);
		}
		for (GRBVar var: model.getVars()){
			int index = varHash.get(var.get(GRB.StringAttr.VarName));
			var.set(GRB.DoubleAttr.Obj, updatedC.getArray()[0][index]);
		}
		model.set(GRB.DoubleAttr.ObjCon, constant);		
	}

	/**
	 * updates the step-size based on the iteration number
	 * @param miu
	 * @param k
	 * @return
	 */
	private static double updateMiu(){
		int itr = (int) Math.floor(k/10);
		miu = miu* Math.pow(0.5,itr);
		return miu;
	}
	
	/**
	 * Concatenates a list of matrices horizontally
	 * @param x
	 * @return
	 */
	private static Matrix concatH(List<Matrix> x){
		int colDim = x.get(0).getColumnDimension(); // column dimension of the output matrix
		int rowDim = 0;	// row dimension of the output matrix
		for (Matrix i:x){
			rowDim += i.getRowDimension();
		}
		Matrix output = new Matrix(rowDim, colDim);
		int i0;
		int i1 = -1;
		for (Matrix i:x){
			i0 = i1+1;
			i1 = i1 + i.getRowDimension();
			output.setMatrix(i0, i1, 0, colDim-1, i);			
		}
		return output;
	}
	
	/**
	 * Updates the miu based on the  rule (c)
	 * @param model
	 * @param varHash
	 * @param UB
	 * @param currentU
	 * @param epsilon
	 * @return
	 * @throws GRBException 
	 */
	private static double updateMiuC(List<Matrix> DList, List<Matrix> dList) throws GRBException{
		// matrices concatenation
		Matrix D = concatH(DList);
		Matrix d = concatH(dList);
		Matrix X = toMatrix(model.getVars(), varHash);
		Matrix denuminatorMat = D.times(X.transpose());		
		denuminatorMat = d.minus(denuminatorMat);
		double denuminator = Math.pow(denuminatorMat.normF(),2);	 
		double output = epsilon*(model.get(GRB.DoubleAttr.ObjVal) - UB ) / denuminator;
		return output;
	}
	
	/**
	 * Updates the Lagrangian Multipliers according to the miu.
	 * @param U
	 * @param d
	 * @param D
	 * @param miu
	 * @param model
	 * @param varHash
	 * @return
	 * @throws GRBException
	 */
	private static Matrix updateU(Matrix U, Matrix d, Matrix D) throws GRBException{
		Matrix X1 = toMatrix(model.getVars(), varHash);
		Matrix output = D.times(X1.transpose());
		
		output = d.minus(output);
		output = U.minus(output.times(miu));
		double[][] u_k = output.getArray();
		for (int i=0;i<u_k.length;i++){
			for (int j=0;j<u_k[0].length;j++){
				u_k[i][j] = (u_k[i][j] > 0) ? u_k[i][j]:0;
			}
		}
		Matrix result = new Matrix(u_k);
		
		return result;
	}
	
	private static Matrix toMatrix(GRBVar[] vars, HashMap<String,Integer> varHash) throws GRBException{
		Matrix output = new Matrix(1, vars.length);
		for (GRBVar var:vars){
			output.set(0, varHash.get(var.get(GRB.StringAttr.VarName)), var.get(GRB.DoubleAttr.X));;			
		}
		return output;
	}
	
	public static void main(String[] arg) throws GRBException{
		GRBEnv env = new GRBEnv("RpHND_LR.log");
		model = new GRBModel(env);
		model.getEnv().set(GRB.IntParam.OutputFlag, 0);
		
		// initializing vars
		GRBVar x1 = model.addVar(0, 1, 4, GRB.BINARY, "x1");
		GRBVar x2 = model.addVar(0, 1, 5, GRB.BINARY, "x2");
		GRBVar x3 = model.addVar(0, 1, 6, GRB.BINARY, "x3");
		GRBVar x4 = model.addVar(0, 1, 7, GRB.BINARY, "x4");
		GRBVar[] vars = {x1,x2,x3,x4};
		model.update();
		
		// creating variables hashMap
		int temp = 0;
		for (GRBVar var : model.getVars()){
			varHash.put(var.get(GRB.StringAttr.VarName),temp++);
		}
		
		// obj function declaration
		GRBLinExpr expr = new GRBLinExpr();
		double[] coEffs = {4,5,6,7};
		expr.addTerms(coEffs, vars);
		model.setObjective(expr, GRB.MAXIMIZE);
		double[][] coEffs0 = {{4,5,6,7}};
		C = new Matrix(coEffs0);		
		
		// constraint 1
		double[][] coEffs1 = {{2,2,3,4}};
		Matrix D1 = new Matrix(coEffs1);
		double[][] rhs1 = {{7}};
		Matrix d1 = new Matrix(rhs1);
		Matrix u1 = new Matrix(1, 1);
		
		//constraint 2
		double[][] coEffs2 = {{1,-1,1,-1}};
		Matrix D2 = new Matrix(coEffs2);
		double[][] rhs2 = {{0}};
		Matrix d2 = new Matrix(rhs2);
		Matrix u2 = new Matrix(1, 1);
		
		/*
		 * Lagrangian Relaxation Algorithm starts:
		 */
//		double secondLastObjVal = -1*GRB.INFINITY;
		model.optimize();
		double lastObjVal = model.get(GRB.DoubleAttr.ObjVal);
		List<Matrix> Ds = new ArrayList<Matrix>();
		List<Matrix> ds = new ArrayList<Matrix>();
		Ds.add(D1); Ds.add(D2);
		ds.add(d1); ds.add(d2);
		
		while(true){
			miu = updateMiu();
//			miu = updateMiuC(Ds, ds);
			u1 = updateU(u1, d1, D1);	// Update the multipliers
			u2 = updateU(u2, d2, D2);
			List<Matrix> Us = new ArrayList<Matrix>();
			Us.add(u1); Us.add(u2);
			updateObjCoeffs(Us, ds, Ds);
			model.optimize();
			k = k + 1;
			printSol();
//			secondLastObjVal = lastObjVal;
			lastObjVal = model.get(GRB.DoubleAttr.ObjVal);
			System.out.println("miu: " + miu + " - u1:" + u1.get(0,0) + " - u2:" + u2.get(0, 0) + " - Itr" + k + ": " + lastObjVal);
		}
		
	}

}
