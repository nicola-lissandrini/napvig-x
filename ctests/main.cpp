#include <iostream>
#include <memory>

/// @brief Define base handle for inherited params
#define NLIB_PARAMS_BASE  \
template<class _DerivedParams>\
const _DerivedParams &params () const {\
	return *dynamic_pointer_cast<_DerivedParams> (_params);\
}

/// @brief Define specific handle for inherited params, shorthand for Base::params<Derived::Params>
#define NLIB_PARAMS_INHERIT(Base) \
const Params &params () const { \
	return Base::params<Params> (); \
}

#define DEF_SHARED(classname) using Ptr = std::shared_ptr<classname>;

using namespace std;

class Terminator
{
public:
	struct Params {
		int a;

		virtual ~Params () {}
		DEF_SHARED(Params)
	};

	template<class _DerivedParams>
	void setParams (const _DerivedParams &params) {
		_params = std::make_shared<_DerivedParams> (params);
	}

protected:
	NLIB_PARAMS_BASE;

private:
	Params::Ptr _params;
};

class CollisionTerminator : public Terminator
{
public:
	struct Params : public Terminator::Params {
		int b;

		DEF_SHARED(Params)
	};

	void dump () {
		cout << "a " << params().a << endl;
		cout << "b " << params().b << endl;
	}

private:
	NLIB_PARAMS_INHERIT(Terminator);
};

int main()
{
	CollisionTerminator bubu;
	CollisionTerminator::Params params;

	params.a = 1;
	params.b = 2;

	bubu.setParams (params);
	bubu.dump();


	return 0;
}
