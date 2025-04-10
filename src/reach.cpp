#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <math.h>

using namespace arma;
using namespace Rcpp;
using namespace std;
using namespace sugar;

// Penalization
arma::mat penal(arma::mat C_ij, double rho, double a, double lw, String penalty){
  arma::mat Cij;
  if (penalty == "L1"){
    Cij = max(0.0,1-lw/rho/norm(C_ij,"fro")) * C_ij;
  }
  else if (penalty == "MCP"){
    if (norm(C_ij,"fro") <= a*lw){
      Cij = max(0.0,1-lw/rho/norm(C_ij,"fro")) * C_ij / (1-1/a/rho);
    }
    else{
      Cij = C_ij;
    }
  }
  else{
    if (norm(C_ij,"fro") <= (1+1/rho)*lw){
      Cij = max(0.0,1-lw/rho/norm(C_ij,"fro")) * C_ij;
    }
    else if (norm(C_ij,"fro") > a*lw){
      Cij = C_ij;
    }
    else{
      Cij = max(0.0,1-lw*a/(a-1)/rho/norm(C_ij,"fro")) * C_ij / (1-1/(a-1)/rho);
    }
  }
  return(Cij);
}

// Row-wise soft-thresholding operator
arma::mat RT(arma::mat A, double lambda) {
  int nrow = A.n_rows;
  int ncol = A.n_cols;
  arma::mat res(nrow, ncol);

  for (int i = 0; i < nrow; i++) {
    double norm_ai = norm(A.row(i),2);
    double factor = 1 - lambda/norm_ai;
    res.row(i) = max(0.0,factor) * A.row(i);
  }
  return res;
}

// Group lasso for Updating U
arma::mat GLasso(arma::mat Y, arma::mat X, arma::mat Z,
                 arma::mat U0, arma::mat V, arma::mat W,
                double lambda, int maxiter){
  int n = Y.n_rows, p = X.n_cols;
  double M = pow(norm(X,2),2);
  arma::mat U = U0, tY = Y - Z*W;

  for (int i = 0; i < maxiter-1; i++){
    U = RT(1/M*X.t()*tY*V+(eye(p,p)-1/M*X.t()*X)*U, n*lambda/M);
  }

  return U;
}

// [[Rcpp::export]]
Rcpp::List reach_Rcpp(arma::mat Y, arma::mat X, arma::mat Z, String penalty,
                      arma::mat U0, arma::mat V0, arma::mat W0,
                      int r, double lb, double lw, double rho, double a,
                      double epsilon, int maxiter,
                      double inner_epsilon, int inner_maxiter)
{
  // Dimensions
  int n = Y.n_rows, q = Y.n_cols, m = int(Z.n_cols/n);
  int i, j, flag = 0;
  // Delta
  arma::mat Delta_ = arma::zeros(n*(n-1)/2,n), Delta;
  for (i = 0; i < n-1; i++){
    for (j = i+1; j < n; j++){
      Delta_(flag,i) = 1;
      Delta_(flag,j) = -1;
      flag++;
    }
  }
  Delta = kron(Delta_,eye(m,m));

  // Initialization
  arma::mat B, U, V, W, Utt, Vtt, Wtt, Ut = U0, Vt = V0, Wt = W0;
  arma::mat C = Delta*W0, C_, Phi = arma::zeros(m*n*(n-1)/2,q);
  // Updates
  int iter, inner_iter;
  double res = 1, inner_res;
  for (iter = 1; iter <= maxiter; iter++){
    if (res < epsilon){
      break;
    }
    inner_res = 1;
    // BCD for U,V,W
    for (inner_iter = 1; inner_iter <= inner_maxiter; inner_iter++){
      if (inner_res < inner_epsilon){
        break;
      }
      // Update U
      Utt = GLasso(Y, X, Z, Ut, Vt, Wt, lb, 10);
      // Update V
      arma::mat tY = Y - Z*Wt, Uw, Vw;
      arma::vec d;
      svd(Uw, d, Vw, tY.t()*X*Utt);
      Vtt = Uw * Vw.t();
      // Update W
      Wtt = (n*rho*Delta.t()*Delta+Z.t()*Z).i() * (Z.t()*(Y-X*Utt*Vtt.t())+n*Delta.t()*(rho*C-Phi));
      inner_res = (pow(norm(Wtt-Wt,"fro"),2)+pow(norm(Utt*Vtt.t()-Ut*Vt.t(),"fro"),2)) / 2;

      Ut = Utt;
      Vt = Vtt;
      Wt = Wtt;
    }
    U = Utt;
    V = Vtt;
    B = U*V.t();
    W = Wtt;

    // Update C
    C_ = Delta*W + Phi/rho;
    for (i = 0; i < n*(n-1)/2; i++){
      C.rows(i*m,(i+1)*m-1) = penal(C_.rows(i*m,(i+1)*m-1),rho,a,lw,penalty);
    }
    // Update Phi
    Phi = Phi + rho*(Delta*W-C);

    res = pow(norm(Delta*W-C,"fro"),2);
  }

  // Group
  int coord = 0;
  arma::vec G = linspace(1,n,n);
  arma::mat c;
  c = C.t();
  c.reshape(m*q,n*(n-1)/2);
  for (i = 0; i < n-1; i++){
    for (j = i+1; j < n; j++){
      if (norm(c.col(coord),2) == 0){
        G(j) = G(i);
      }
      coord++;
    }
  }

  arma::vec temp = unique(G);
  arma::uvec ind;
  int K = temp.n_elem;
  for (i = 0; i < K; i++){
    ind = find(G == temp(i));
    G.elem(ind) = ones<vec>(ind.n_elem) * int(i+1);
  }

  // GIC
  double GIC;
  int s;
  s = sum(mean(B,1)!=0);
  GIC = log(pow(norm(Y-X*B-Z*W,"fro"),2)/(n*q)) + 15*(log(log(n*q))/(n*q))*(K*m*q+(s+q-r)*r);

  // Output
  Rcpp::List out;
  out["B"] = B;
  out["U"] = U;
  out["V"] = V;
  out["W"] = W;
  out["C"] = C;
  out["iter"] = iter - 1;
  out["res"] = res;
  out["G"] = G;
  out["K"] = K;
  out["GIC"] = GIC;
  out["s"] = s;
  return(out);
}
