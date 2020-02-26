// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>



using namespace Rcpp;
using namespace arma;


Rcpp::NumericVector arma2vec(arma::vec x) {
  return Rcpp::NumericVector(x.begin(), x.end());
}

// [[Rcpp::export]]
arma::mat cpp_normal_interpolation(arma::vec t, arma::vec alpha){
  // Basic Boostrap Confidence Method
  
  // if alpha equal to 1, then substitute to 0.999
  alpha.replace(1, .999);
  
  
  arma::vec finite_t = t.elem(find_finite(t));
  int R = finite_t.n_elem;
  arma::vec rk = (R+1) * alpha;
  arma::vec k = trunc(rk);
  
  
  // values of k that are not 0 or R
  arma::uvec ind_kvs = find(k > 0 and k < R);
  arma::vec kvs = k.elem(ind_kvs);
  
  // sorted t
  arma::vec sorted_t = sort(t);
  
  // output
  arma::vec out(alpha.n_elem);
  
  // find integers
  arma::uvec int_rk = find(k == rk);
  if(int_rk.n_elem > 0){
    arma::uvec ind_sorted_t = conv_to<uvec>::from(k.elem(int_rk));
    out.elem(int_rk) = sorted_t.elem(ind_sorted_t);
  }
  
  // Find k==0 or k == R
  arma::uword Ri = R;
  out.elem(find(k == 0)).fill(sorted_t(0));
  out.elem(find(k == R)).fill(sorted_t(Ri-1));
  
  // find non-interger, non-0 and non-R
  arma::uvec ind_not = find(k != rk and k > 0 and k < R);
  
  // interpolate
  arma::vec temp1 = Rcpp::qnorm(arma2vec(alpha.elem(ind_not)));
  arma::vec temp2 = Rcpp::qnorm(arma2vec(k.elem(ind_not)/(R+1)));
  arma::vec temp3 = Rcpp::qnorm(arma2vec((k.elem(ind_not) + 1)/(R+1)));
  
  // tks
  arma::uvec ind_tk = conv_to<uvec>::from(k.elem(ind_not) - 1);
  arma::uvec ind_tk1 = conv_to<uvec>::from(k.elem(ind_not));
  arma::vec tk = sorted_t(ind_tk);
  arma::vec tk1 = sorted_t(ind_tk1);
  
  // complete output
  out.elem(ind_not) = tk + (temp1-temp2)/(temp3-temp2)%(tk1 - tk);

  // output matrix
  arma::mat out_mat = join_rows(rk, out);
  
  return out_mat;
}




// [[Rcpp::export]]
arma::mat cpp_boot_basic_ci(arma::mat t0, arma::cube t, arma::vec conf){
  // output matrix
  arma::mat mat_ci_output(t.n_cols * t.n_slices * conf.n_elem, 7);
  
  // creating alpha
  arma::mat alpha(conf.n_rows, 2);
  arma::vec uns(conf.n_elem, fill::ones);
  alpha.col(0) = (uns + conf)/2;
  alpha.col(1) = (uns - conf)/2;
  arma::vec alpha_vec = vectorise(alpha);
  
  // calcualting interpolation
  uword r = 0;
  for(uword s = 0; s < t.n_slices; ++s){
    for(uword j = 0; j < t.n_cols; ++j){
      arma::mat qq = cpp_normal_interpolation(t.slice(s).col(j), alpha_vec);
      for(uword i = 0; i < conf.n_elem; ++ i){
        mat_ci_output(r, 0) = s + 1;
        mat_ci_output(r, 1) = j + 1;
        mat_ci_output(r, 2) = conf(i);
        arma::uvec qq_row_idx = {i, i + conf.n_elem};
        arma::uvec qq_col_idx(1, fill::zeros);
        mat_ci_output.row(r).cols(3, 4) = qq(qq_row_idx, qq_col_idx).t();
        mat_ci_output.row(r).cols(5, 6) = 2*t0(j,s) - qq(qq_row_idx, qq_col_idx + 1).t();
        r++;
      }
    }
  }
  
  return mat_ci_output; 
}

// [[Rcpp::export]]
arma::mat cpp_boot_perc_ci(arma::mat t0, arma::cube t, arma::vec conf){
  // output matrix
  arma::mat mat_ci_output(t.n_cols * t.n_slices * conf.n_elem, 7);
  
  // creating alpha
  arma::mat alpha(conf.n_rows, 2);
  arma::vec uns(conf.n_elem, fill::ones);
  alpha.col(0) = (uns - conf)/2;
  alpha.col(1) = (uns + conf)/2;
  arma::vec alpha_vec = vectorise(alpha);
  
  // calcualting interpolation
  uword r = 0;
  for(uword s = 0; s < t.n_slices; ++s){
    for(uword j = 0; j < t.n_cols; ++j){
      arma::mat qq = cpp_normal_interpolation(t.slice(s).col(j), alpha_vec);
      for(uword i = 0; i < conf.n_elem; ++ i){
        mat_ci_output(r, 0) = s + 1;
        mat_ci_output(r, 1) = j + 1;
        mat_ci_output(r, 2) = conf(i);
        arma::uvec qq_row_idx = {i, i + conf.n_elem};
        arma::uvec qq_col_idx(1, fill::zeros);
        mat_ci_output.row(r).cols(3, 4) = qq(qq_row_idx, qq_col_idx).t();
        mat_ci_output.row(r).cols(5, 6) = qq(qq_row_idx, qq_col_idx + 1).t();
        r++;
      }
    }
  }
  
  return mat_ci_output; 
}

// [[Rcpp::export]]
arma::vec cpp_empirical_influence_reg(arma::vec t, arma::mat boot_array, int n){
  /* Function to estimate empirical influence values using regression.
   This method regresses the observed bootstrap values on the bootstrap
   frequencies to estimate the empirical influence values
   */
  
  arma::uvec fins = find_finite(t);
  arma::vec finite_t = t.elem(fins);
  int R = finite_t.n_elem;
  
  // if strata is null
  arma::vec strata(n, fill::ones);
  
  
  // boot array
  arma::mat finite_boot_array = boot_array.rows(fins);
  
  arma::mat X(R, n);
  X.fill(strata.n_elem);
  X = finite_boot_array / X;
  
 
  // output
  uword out = 0;
  arma::uvec inc = conv_to<uvec>::from(linspace(0, n-1, n));
  inc.shed_row(out);
  X = X.cols(inc);
  arma::vec uns(X.n_rows, fill::ones);
  X = join_rows(uns, X);
  arma::vec beta = inv(X.t()*X) * X.t() * t;
  beta.shed_row(0);
  arma::vec l(n, fill::zeros);
  l.elem(inc) = beta;
  l = l - mean(l);
  
  
  return l;
  
}

// [[Rcpp::export]]
arma::vec cpp_empirical_influence(arma::vec t, arma::mat boot_array, int n){
  /* Calculation of empirical influence values.  Possible types are
    0: "inf" = infinitesimal jackknife (numerical differentiation)
    1: "reg" = regression based estimation
    2: "jack" = usual jackknife estimates
    3: "pos" = positive jackknife estimates */

  arma::vec L = cpp_empirical_influence_reg(t, boot_array, n);

  return L;
}



// [[Rcpp::export]]
arma::mat cpp_boot_bca_ci(arma::mat t0, arma::cube t, arma::vec conf, arma::mat boot_array, int n){
  // output matrix
  arma::mat mat_ci_output(t.n_cols * t.n_slices * conf.n_elem, 7);
  
  
  
  
  
  // creating alpha
  arma::mat alpha(conf.n_rows, 2);
  arma::vec uns(conf.n_elem, fill::ones);
  alpha.col(0) = (uns - conf)/2;
  alpha.col(1) = (uns + conf)/2;
  
  
  // creating zalpha
  arma::mat zalpha = alpha;
  arma::vec temp0 = Rcpp::qnorm(arma2vec(zalpha.col(0)));
  arma::vec temp1 = Rcpp::qnorm(arma2vec(zalpha.col(1)));
  zalpha.col(0) = temp0;
  zalpha.col(1) = temp1;
  

  
  // calcualting interpolation
  uword r = 0;
  for(uword s = 0; s < t.n_slices; ++s){
    for(uword j = 0; j < t.n_cols; ++j){
      
      // original t and finite_t
      arma::vec t_o = t.slice(s).col(j);
      arma::uvec finite_t_indices = find_finite(t_o);
      arma::vec finite_t = t_o.elem(finite_t_indices);
       
      
      // w, 
      double tot_menor_t0 = accu(finite_t < t0(j, s));
      double tot_t = finite_t.n_elem;
      double w = tot_menor_t0/tot_t;
      
      arma::vec vec_w(1);
      vec_w.fill(w);
      vec_w = Rcpp::qnorm(arma2vec(vec_w));
      w = as_scalar(vec_w);
      
      // a
      arma::vec L = cpp_empirical_influence(t_o, boot_array, n);
      double a = accu(pow(L, 3)) / (6*pow(accu(pow(L, 2)), 1.5));
    
      // adjusted alpha
      arma::mat adj_alpha = normcdf(w + (w + zalpha)/(1 - (a*(w + zalpha))));
      arma::vec adj_alpha_vec = vectorise(adj_alpha);
      
      arma::mat qq = cpp_normal_interpolation(t_o, adj_alpha_vec);
      for(uword i = 0; i < conf.n_elem; ++ i){
        mat_ci_output(r, 0) = s + 1;
        mat_ci_output(r, 1) = j + 1;
        mat_ci_output(r, 2) = conf(i);
        arma::uvec qq_row_idx = {i, i + conf.n_elem};
        arma::uvec qq_col_idx(1, fill::zeros);
        mat_ci_output.row(r).cols(3, 4) = qq(qq_row_idx, qq_col_idx).t();
        mat_ci_output.row(r).cols(5, 6) = qq(qq_row_idx, qq_col_idx + 1).t();
        r++;
      }
    }
  }
  
  return mat_ci_output; 
}


// [[Rcpp::export]]
arma::mat cpp_boot_norm_ci(arma::mat t0, arma::cube t, arma::vec conf){
  // output matrix
  arma::mat mat_ci_output(t.n_cols * t.n_slices * conf.n_elem, 7);
  

  // calcualting interpolation
  uword r = 0;
  for(uword s = 0; s < t.n_slices; ++s){
    for(uword j = 0; j < t.n_cols; ++j){
      
      // original t and finite_t
      arma::vec t_o = t.slice(s).col(j);
      arma::uvec finite_t_indices = find_finite(t_o);
      arma::vec finite_t = t_o.elem(finite_t_indices);
      
      // variance of t0
      double var_t0 = var(finite_t);
      
      // bias of t
      double bias = mean(finite_t) - t0(j, s);
      
      // erro padrão
      arma::vec errop = sqrt(var_t0) * Rcpp::qnorm(arma2vec((1 + conf)/2));
      
      for(uword i = 0; i < conf.n_elem; ++ i){
        mat_ci_output(r, 0) = s + 1;
        mat_ci_output(r, 1) = j + 1;
        mat_ci_output(r, 2) = conf(i);
        mat_ci_output(r, 3) = datum::nan;
        mat_ci_output(r, 4) = datum::nan;
        mat_ci_output(r, 5) = t0(j, s) - bias - errop(i);
        mat_ci_output(r, 6) = t0(j, s) - bias + errop(i);
        r++;
      }
    }
  }
  
  return mat_ci_output; 
}



//arma::mat cpp_boot_stud_ci(arma::mat t0, arma::cube t, arma::vec conf, arma::mat var_t0){
//  // output matrix
//  arma::mat mat_ci_output(t.n_cols * t.n_slices * conf.n_elem, 7);
//  
//  // creating alpha
//  arma::mat alpha(conf.n_rows, 2);
//  arma::vec uns(conf.n_elem, fill::ones);
//  alpha.col(0) = (uns + conf)/2;
//  alpha.col(1) = (uns - conf)/2;
//  arma::vec alpha_vec = vectorise(alpha);
//  
//  // calcualting interpolation
//  uword r = 0;
//  for(uword s = 0; s < t.n_slices; ++s){
//    for(uword j = 0; j < t.n_cols; ++j){
//      
//      arma::vec z = (t.slice(s).col(j) - t0)/
//      
//      arma::mat qq = cpp_normal_interpolation(t.slice(s).col(j), alpha_vec);
//      for(uword i = 0; i < conf.n_elem; ++ i){
//        mat_ci_output(r, 0) = s + 1;
//        mat_ci_output(r, 1) = j + 1;
//        mat_ci_output(r, 2) = conf(i);
//        arma::uvec qq_row_idx = {i, i + conf.n_elem};
//        arma::uvec qq_col_idx(1, fill::zeros);
//        mat_ci_output.row(r).cols(3, 4) = qq(qq_row_idx, qq_col_idx).t();
//        mat_ci_output.row(r).cols(5, 6) = 2*t0(j,s) - qq(qq_row_idx, qq_col_idx + 1).t();
//        r++;
//      }
//    }
//  }
//   
//   return mat_ci_output; 
// }