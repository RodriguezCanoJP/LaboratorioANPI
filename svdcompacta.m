function [Ur, Sr, Vr]=svdcompacta(A)

  [m,n]=size(A);
  if m>n
    M1 = A'*A;
    [V1, D]=eig(M1); %V1 -> vector propio, D-> valores propios
    y1=diag(D); %valores propios en matriz diagonal
    const=n*max(y1)*eps; %constante para comparar valores propios
    y2=(y1>const); %matriz que retorna cuantos valores cumplen con la condicion
    rA=sum(y2); %rango de la matriz
    y3=y1.*y2;
    [s1, order]=sort(sqrt(y1), 'descend');
    V2=V1(:,order);
    Vr=V2(:,1:rA);
    Sr=diag(s1(1:rA));
    Ur=(1./(s1(1:rA))').*(A*Vr);

  else
    M1=A*A';
    [U1, D]=eig(M1);
    y1=diag(D);
    const=m*max(y1)*eps;
    y2=(y1>const);
    rA=sum(y2);
    y3=y1.*y2;
    [s1, order]=sort(sqrt(y3), 'descend');
    U2=U1(:, order);
    Ur=U2(:, 1:rA);
    Sr=diag(s1(1:rA));
    Vr=(1./(s1(1:rA))').*(A'*Ur);
  endif
endfunction



