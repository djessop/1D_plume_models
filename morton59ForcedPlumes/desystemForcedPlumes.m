function Qdot=desystemForcedPlumes( z,Q )

% System of differential equations as defined by Morton (1959) for forced plumes
%   dq/dz=m^{1/2}
%   dm/dz=f*q/(4*m^3)
%   df/dz=-q

Qdot=[	Q(2);                                   % dQ/dz
        Q(1)*Q(3)/( 4.0*Q(2).^3 );              % dM/dz
	   -Q(1)	];								% dF/dz
