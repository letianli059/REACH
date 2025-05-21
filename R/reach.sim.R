#' @title Simulate heterogeneous sparse reduced-rank regressions
#'
#' @description
#' Similar to the models in simulation examples 1-2 in Li et al. (2025)
#'
#' @param n,p,m,q model dimensions
#' @param r model rank
#' @param s model sparsity level
#' @param d_Theta parameter of the distribution for generating \eqn{\Theta_k}
#' @param rho_X correlation parameter in the generation of covariates
#' @param sigma_X variance parameter in the generation of covariates
#' @param s2n signal to noise ratio
#'
#' @return similated model and data
#'
#' @importFrom stats rnorm rbinom
#' @export
reach.sim <- function(n, p, q, m, r, s, d_Theta = 5, rho_X = .5, sigma_X = 1, s2n = 2){
  # Generate X
  Sigma_X <- matrix(nrow = p, ncol = p, rho_X)
  diag(Sigma_X) <- sigma_X
  X0 <- MASS::mvrnorm(max(n,p), rep(0,p), Sigma_X, empirical = TRUE)[1:n,]
  X <- X0
  # Generate Z_
  if (m > 1){
    Z_ <- matrix(rbinom((m-1)*n,1,.7), n)
    Z_ <- cbind(rep(1,n),Z_)
  }else{
    Z_ <- matrix(1, n ,1)
  }
  # Generate Z
  Z <- matrix(0, n, m*n)
  for (i in 1:n){
    Z[i,((i-1)*m+1):(i*m)] <- Z_[i,]
  }
  # Generate B
  B1 <- matrix(rnorm(s*r,0,sigma_X), nrow = s, ncol = r)
  B2 <- matrix(rnorm(r*q,0,sigma_X), nrow = r, ncol = q)
  B <- rbind(B1%*%B2, matrix(0,nrow = p-s,ncol = q))
  # Generate W
  #Theta1 <- matrix(d_Theta, m, q)
  Theta1 <- matrix(runif(m*q,0,d_Theta), m, q)
  #Theta1 <- matrix(rnorm(m*q,d_Theta,1), m, q)
  Theta2 <- -Theta1
  Theta3 <- matrix(rep(0,m*q), m, q)
  Theta <- list(Theta1,Theta2,Theta3)
  W <- matrix(0, nrow = m*n, ncol = q)
  groups <- sample(1:3, n, replace = T)
  for (i in 1:n){
    W[((i-1)*m+1):(i*m),] <- Theta[[groups[i]]]
  }
  # Generate E
  SigmaE <- matrix(0, nrow = q, ncol = q)
  diag(SigmaE) <- sigma_X
  E <- MASS::mvrnorm(max(n,q), rep(0,q), SigmaE, empirical = TRUE)[1:n,]
  rd.XB <- svd(X%*%B)$d[r]
  sigma <- rd.XB / (norm(E,"F")*s2n)
  # Generate Y
  Y = X%*%B + Z%*%W + sigma*E

  # Output
  list(Y = Y, X = X, B = B, Z_ = Z_, Z = Z, W = W, Theta = Theta, group = groups)
}
