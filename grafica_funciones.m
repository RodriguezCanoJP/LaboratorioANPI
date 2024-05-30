function grafica_funciones()
  clc;
  x = 5:12;
  y_svd_octave = zeros(1,length(x));
  y_svd_compacta = zeros(1,length(x));
  idx = 1;
  for i = x
    A=randi(i,i-1);
    tic();
    svd(A);
    y_svd_octave(idx) = toc();
    tic();
    svdcompacta(A);
    y_svd_compacta(idx) = toc();
    idx = idx + 1;
  endfor

  figure;
  hold on;
  plot(x, y_svd_compacta, 'r-', 'LineWidth', 2);
  plot(x, y_svd_octave, 'b-', 'LineWidth', 2);
  xlabel('Tamaño de matriz', 'FontSize', 12);
  ylabel('Tiempo', 'FontSize', 12);
  title('Tiempo de ejecución de métodos de SVD', 'FontSize', 14);
  legend('SVD Compacta', 'GNU Octave');
  grid on;
  hold off;
endfunction
