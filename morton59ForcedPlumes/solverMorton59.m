function [z,Q]=solverMorton59( varargin )

% SOLVERMORTON59 Solves the system of ODEs for a forced plume.  The model is
%	physically identical to that of MTT56.
%
% INPUTS (optional)
%   q0  initial volume flux
%   m0  initial momentum flux
%   f0  initial buoyancy flux
%
% CALLS ON:
%	MORTON59EQNS: the system of equations to be solved.
%
% DEJESSOP, 20th June 2013

global eta alpha lambda G q0 m0 f0 Z

% INTITIAL CONDITIONS

if ( nargin==0 ); 
    eta=1e-3;
    q0=eta;
    m0=1;
    f0=1;
    alpha = 0.1;
    lambda = 1.1;
    G = .012^2; % typically, N = 0.012 s^{-1} in the atmosphere
    Z = 2;
elseif ( nargin==3 );
    [q0,m0,f0] = deal( varargin{:} );
elseif ( nargin==4 );
    [q0,m0,f0,Z] = deal( varargin{:} );
else error( 'incorrect number of inputs' );
end

Q0=[ q0 m0 f0 ]';

% RANGE OF SOLUTION
z=linspace( 0,Z,501 )';

opts = odeset( 'Stats','off' );

% SOLVER
[z,Q]=ode23s( @morton59eqns,z,Q0,opts );
