#pragma once

#include "GlobalDefs.h"
#include <wx/textfile.h>

using namespace std;

/****************************************************************************************/
/*	struct StringHash																	*/
/*																						*/
/*	Defines the comparison and key value functions needed to use std::string's in		*/
/*	a hash_map.																			*/
/****************************************************************************************/
typedef struct STRINGHASH_STRUCT
{
	enum
	{	// parameters for hash table
		bucket_size = 4,	// 0 < bucket_size
		min_buckets = 8
	};	// min_buckets = 2 ^^ N, 0 < N

	size_t operator()(const string& s) const
	{	// hash _Keyval to size_t value
		const unsigned char *p = (const unsigned char *)s.c_str();

		size_t hashval = 0;

		while (*p != '\0') {
			hashval += *p++;
		}

		return hashval;
	}

	bool operator()(const string &s1, const string &s2) const
	{ // test if s1 ordered before s2
		return (s1 < s2);
	} 

} StringHash;


/****************************************************************************************/
/*	struct ParameterElement																*/
/*																						*/
/*	Defines the element associated with a given command key.							*/
/****************************************************************************************/
typedef struct PARAMVAL_STRUCT
{
	string description;
	vector <double>data;
	bool variable;
} ParameterValue;

// These help us to be lazy.
typedef pair <const string, ParameterValue> ParameterKeyPair;
typedef hash_map <string, ParameterValue, StringHash> ParameterHash;
typedef hash_map<string, ParameterValue>::iterator ParameterIterator;


class CParameterList
{
private:
	ParameterHash m_pHash;

public:
	CParameterList();

	// Returns a pointer to a list of all the keys in the hash.
	string * GetKeyList(int &keyCount);

	// Returns the number of keys in the hash.
	int GetListSize() const;

	// Returns the description of a given parameter.
	string GetParamDescription(string param);

	// Returns the size of a given parameter.
	int GetParamSize(string param);

	// Gets/Sets the vector data associated with a specified key.
	vector<double> GetVectorData(string key);
	void SetVectorData(string key, vector<double> value);

	// Checks to see if a parameter exists in the hashtable.
	bool Exists(string key);

	// Determines if the parameter is variable in size.
	bool IsVariable(string param);

private:
	// Loads the parameter list with all the hash values.  This is where you'd put all
	// hash keys and values and set defaults.
	void LoadHash();
};
