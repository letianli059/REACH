#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <math.h>

using namespace arma;
using namespace Rcpp;
using namespace std;
using namespace sugar;

// Row-wise soft-thresholding operator
arma::mat RT_ridge(arma::mat A, double lambda) {
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
arma::mat GLasso_ridge(arma::mat Y, arma::mat X, arma::mat Z,
                 arma::mat U0, arma::mat V, arma::mat W,
                 double lambda, int maxiter){
  int n = Y.n_rows, p = X.n_cols;
  double M = pow(norm(X,2),2);
  arma::mat U = U0, tY = Y - Z*W;

  for (int i = 0; i < maxiter-1; i++){
    U = RT_ridge(1/M*X.t()*tY*V+(eye(p,p)-1/M*X.t()*X)*U, n*lambda/M);
  }

  return U;
}

// [[Rcpp::export]]
Rcpp::List ridge_Rcpp(arma::mat Y, arma::mat X, arma::mat Z,
                      arma::mat U0, arma::mat V0, arma::mat W0,
                      int r, double lb, double lw, double a,
                      double epsilon, int maxiter)
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
  arma::mat U, V, W, Utt, Vtt, Wtt, Ut = U0, Vt = V0, Wt = W0;
  // Updates
  int iter;
  double res = 1;
  for (iter = 1; iter <= maxiter; iter++){
    if (res < epsilon){
      break;
    }
    // Update U
    Utt = GLasso_ridge(Y, X, Z, Ut, Vt, Wt, lb, 10);
    // Update V
    arma::mat tY = Y - Z*Wt, Uw, Vw;
    arma::vec d;
    svd(Uw, d, Vw, tY.t()*X*Utt);
    Vtt = Uw.cols(0,r-1) * Vw.t();
    // Update W
    Wtt = (n*lw*Delta.t()*Delta+Z.t()*Z).i() * Z.t() * (Y-X*Utt*Vtt.t());
    res = (pow(norm(Wtt-Wt,"fro"),2)+pow(norm(Utt*Vtt.t()-Ut*Vt.t(),"fro"),2)) / 2;

    Ut = Utt;
    Vt = Vtt;
    Wt = Wtt;
  }

  U = Utt;
  V = Vtt;
  W = Wtt;

  // Output
  Rcpp::List out;
  out["U"] = U;
  out["V"] = V;
  out["W"] = W;
  return(out);
}
