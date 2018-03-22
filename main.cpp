#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>

#include "timer.h"
#include "matrix.h"




int main()
{
	constexpr auto m = 2000u, n = 2000u;
	using type    = double;
	using storage = cmat::storage::row_major;
	using matrix  = cmat::matrix<type,m,n,storage>;
	using timer   = cmat::timer<cmat::milliseconds>;

	auto J = matrix{},
	     B = matrix{};

	for(auto ri = 0u, j = 0u, k = 0u; ri < J.rows(); ++ri)
		for(auto ci = 0u; ci < J.cols(); ++ci)
			J.at(ri,ci) = k++, B.at(ri,ci) = j++;

	timer::before();
	matrix D = J|B;
	timer::after();
	std::cout << "Expression[JxB], " << "Time[ms]: "<< timer::elaps() << std::endl;

	timer::before();
	matrix E = 6.0 * J - B + J*B - 2.0*D;
	timer::after();
	std::cout << "Expression[2.0 * J - B + B- 4.0*D], " << "Time[ms]: "<< timer::elaps() << std::endl;

	timer::before();
	matrix F = 2.0 * (J|B)  + B - 4.0*D + 3.0*E;
	timer::after();
	std::cout << "Expression[2.0 * JxB  + B - D- 4.0*D + 3.0*E], " << "Time[ms]: "<< timer::elaps() << std::endl;



	if(m<15 && n<15)
	{
		std::ofstream out("check.m");
		out << std::setprecision(std::numeric_limits<type>::max_digits10);
		out << "J=" << J << std::endl;
		out << "B=" << B << std::endl;
		out << "D=" << D << std::endl;
		out << "E=" << E << std::endl;
		out << "F=" << F << std::endl;
		out << "Dref = J*B;" << std::endl;
		out << "Eref = 2.*J.-B.+B.-4.*Dref;" << std::endl;
		out << "Fref = 2.*J*B.+B.-4.*Dref.+3.*Eref;" << std::endl;
		out << "printf('MaxAbsErr(D) = %f\\n',max(D(:)-Dref(:)));" << std::endl;
		out << "printf('MaxAbsErr(E) = %f\\n',max(E(:)-Eref(:)));" << std::endl;
		out << "printf('MaxAbsErr(F) = %f\\n',max(F(:)-Fref(:)));" << std::endl;

	}
}
