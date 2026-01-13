function dQdx = morton59eqns( x,Q )
%MORTON59EQNS   ODE system for forced plumes as described by Morton 1959
% Uniform environment:
%   dW/dX = 2^{3/2} * alpha * V, 
%   dV/dX = (1+lambda^2)*F*W/(4*V^3)
%   dF/dX = 0
% Transformations: 
%   F = F0*f
%   V = abs(V0)*v,
%   W = 2^.75*sqrt(alpha)*(1+lambda^2)^-.5*abs(V0)^2.5*abs(F0)^-.5w.
%
% STABLY STRATIFIED ENVIRONMENT:
%   dW/dX = 2 * alpha * V, 
%   dV/dX = 2*lambda*F*W/(4*V^3)
%   dF/dX = -G*W
% Transformations
%   F = abs(F0)*f
%   V = 2^.25*lambda^.5*abs(F0)^.5^G^-.25*v
%   W = 2^.625*alpha^.5*lambda*.25*abs(F0)^.75*G^-.625*w
%   X = 2^-.625*alpha^.-5*lambda^-.25*G^-.375*x
% Dimensionless equations
%   dw/dx = v, dv/dx = f*w/(4*v^3), df/dx = -w

% global alpha lambda
% global f0

w = Q(1);
v = Q(2);
f = Q(3);
% 
% % Dimensional equations
% global alpha lambda G
% 
% dQdx = [ 2^1.5 * alpha * v;
%     (1+lambda^2)*f*w/(4*v^3);
%     -G*w ];

% Dimensionless equations
dQdx = [ v;
    w * f / ( 4*v^3 );
    -w ];

end

