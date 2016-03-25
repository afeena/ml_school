#include <mysql++/mysql++.h>
#include <mysql++/query.h>
#include <mysql++/connection.h>
#include <vector>
#include <string>
class DB {
public:
	DB();
	virtual ~DB();
	void save(std::vector<int> identificators, std::vector<int> labels);
private:
	mysqlpp::Connection connection;

};


