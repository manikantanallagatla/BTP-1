function CC = bwconncomp (bw, N)

  if (nargin < 1 || nargin > 2)
    print_usage ();
  elseif (! ismatrix (bw) || ! (isnumeric (bw) || islogical (bw)))
    error ("bwconncomp: BW must be an a numeric matrix");
  endif
  if (nargin < 2)
    ## Defining default connectivity here because it's dependent
    ## on the first argument
    N = conndef (ndims (bw), "maximal");
  endif
  [conn, N] = make_conn ("bwconncomp", 2, ndims (bw), N);

  [bw, n_obj] = bwlabeln (logical (bw), conn);
  ## We should probably implement this as the first part of bwlabeln
  ## as getting the indices is the first part of its code. Here we are
  ## just reverting the work already done.
  P = arrayfun (@(x) find (bw == x), 1:n_obj, "UniformOutput", false);

  ## Return result
  CC = struct ("Connectivity",  N,
               "ImageSize",     size (bw),
               "NumObjects",    n_obj,
               "PixelIdxList",  {P});
endfunction