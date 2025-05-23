// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// reach_Rcpp
Rcpp::List reach_Rcpp(arma::mat Y, arma::mat X, arma::mat Z, String penalty, arma::mat U0, arma::mat V0, arma::mat W0, int r, double lb, double lw, double rho, double a, double epsilon, int maxiter, double inner_epsilon, int inner_maxiter);
RcppExport SEXP _reach_reach_Rcpp(SEXP YSEXP, SEXP XSEXP, SEXP ZSEXP, SEXP penaltySEXP, SEXP U0SEXP, SEXP V0SEXP, SEXP W0SEXP, SEXP rSEXP, SEXP lbSEXP, SEXP lwSEXP, SEXP rhoSEXP, SEXP aSEXP, SEXP epsilonSEXP, SEXP maxiterSEXP, SEXP inner_epsilonSEXP, SEXP inner_maxiterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< String >::type penalty(penaltySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type U0(U0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type V0(V0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type W0(W0SEXP);
    Rcpp::traits::input_parameter< int >::type r(rSEXP);
    Rcpp::traits::input_parameter< double >::type lb(lbSEXP);
    Rcpp::traits::input_parameter< double >::type lw(lwSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type epsilon(epsilonSEXP);
    Rcpp::traits::input_parameter< int >::type maxiter(maxiterSEXP);
    Rcpp::traits::input_parameter< double >::type inner_epsilon(inner_epsilonSEXP);
    Rcpp::traits::input_parameter< int >::type inner_maxiter(inner_maxiterSEXP);
    rcpp_result_gen = Rcpp::wrap(reach_Rcpp(Y, X, Z, penalty, U0, V0, W0, r, lb, lw, rho, a, epsilon, maxiter, inner_epsilon, inner_maxiter));
    return rcpp_result_gen;
END_RCPP
}
// ridge_Rcpp
Rcpp::List ridge_Rcpp(arma::mat Y, arma::mat X, arma::mat Z, arma::mat U0, arma::mat V0, arma::mat W0, int r, double lb, double lw, double a, double epsilon, int maxiter);
RcppExport SEXP _reach_ridge_Rcpp(SEXP YSEXP, SEXP XSEXP, SEXP ZSEXP, SEXP U0SEXP, SEXP V0SEXP, SEXP W0SEXP, SEXP rSEXP, SEXP lbSEXP, SEXP lwSEXP, SEXP aSEXP, SEXP epsilonSEXP, SEXP maxiterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type U0(U0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type V0(V0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type W0(W0SEXP);
    Rcpp::traits::input_parameter< int >::type r(rSEXP);
    Rcpp::traits::input_parameter< double >::type lb(lbSEXP);
    Rcpp::traits::input_parameter< double >::type lw(lwSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type epsilon(epsilonSEXP);
    Rcpp::traits::input_parameter< int >::type maxiter(maxiterSEXP);
    rcpp_result_gen = Rcpp::wrap(ridge_Rcpp(Y, X, Z, U0, V0, W0, r, lb, lw, a, epsilon, maxiter));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_reach_reach_Rcpp", (DL_FUNC) &_reach_reach_Rcpp, 16},
    {"_reach_ridge_Rcpp", (DL_FUNC) &_reach_ridge_Rcpp, 12},
    {NULL, NULL, 0}
};

RcppExport void R_init_reach(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
