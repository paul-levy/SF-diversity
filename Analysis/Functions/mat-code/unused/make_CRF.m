function crf = make_CRF(c50, n)

crf = @(con) (con.^n) ./ (c50 + con.^n);

end