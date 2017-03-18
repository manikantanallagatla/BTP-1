function imout = im2uint8 (img, varargin)
  if (nargin < 1 || nargin > 2)
    print_usage ();
  elseif (nargin == 2 && ! strcmpi (varargin{1}, "indexed"))
    error ("im2uint8: second input argument must be the string \"indexed\"");
  endif
  imout = imcast (img, "uint8", varargin{:});
endfunction