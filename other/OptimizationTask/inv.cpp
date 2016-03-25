#include <unordered_map>
#include <unordered_set>
#include <bitset>
#include <iostream>
#include <cmath>
#include <memory>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <map>
#include <cassert>
#include <set>
#include <random>
#include <vector>
#include <functional>
#include <sstream>
#include <bitset>
#include <iterator>
#include <climits>

using namespace std;

random_device rd;
std::mt19937 rng(rd());
std::uniform_int_distribution<int> uni20_50(20, 50);
std::uniform_int_distribution<int> uni0_100000(0, 100000);

struct Document{
    int id;
    vector<int> words;
    vector<bool> binaryDescriptor;

    static Document generate_document(int id, int document_length = uni20_50(rng))
    {
        Document doc;
        doc.id = id;

        for (int i = 0; i < document_length; i++)
            doc.words.push_back(uni0_100000(rng));

        for (int i = 0; i < 256; i++)
            doc.binaryDescriptor.push_back(uni0_100000(rng) > 50000);

        return doc;
    }
};
using Corpus = vector<Document>;

class Index
{
public:
    struct SearchResult
    {
        int documentID;
        int distance;

        SearchResult()
        {
            SearchResult(-1, -1);
        }

        SearchResult(int documentID, int distance)
            :documentID(documentID), distance(distance)
        {
        }

        bool operator<(const SearchResult &result) const
        {
            return make_pair(this->distance, this->documentID) < make_pair(result.distance, result.documentID);
        }
    };

    virtual ~Index(){}
    virtual SearchResult search(const Document &document) const = 0;
};

class SimpleIndex : public Index{
public:
    SimpleIndex(const Corpus &corpur)
    {
        for (Document document : corpur)
        {
            for (auto word : document.words)
                invertedIndex[word].push_back(document.id);
            imageDescriptors[document.id] = document.binaryDescriptor;
        }
    }

    SearchResult search(const Document &document) const
    {
        SearchResult result;

        if (document.words.empty())
            return result;

        set<int> candidatesSet;
        for (auto word : document.words)
        {
            if (invertedIndex.find(word) != invertedIndex.end())
                for (auto documentID : invertedIndex.at(word))
                    candidatesSet.insert(documentID);
        }

        vector<SearchResult> documentDistances;
        for (auto documentID : candidatesSet)
        {
            int distance = hammingDistance(document.binaryDescriptor, imageDescriptors.at(documentID));
            documentDistances.push_back(SearchResult(documentID, distance));
        }
        sort(documentDistances.begin(), documentDistances.end());

        if (documentDistances.size())
            result = documentDistances.front();

        return result;
    }

    static int hammingDistance(const vector<bool> &a, const vector<bool> &b)
    {
        int distance = 0;
        for (size_t i = 0; i < a.size(); i++)
            if (a[i] != b[i])
                distance++;
        return distance;
    }

private:
    map<int, vector<int>> invertedIndex;
    map<int, vector<bool>> imageDescriptors;
};

class OptimizedIndex : public Index{
public:
    OptimizedIndex(const Corpus &corp)
    {
        for (Document document : corp)
        {
            bitset<256> bs;
            for (size_t i = 0; i < document.binaryDescriptor.size(); i++)
                bs[i] = document.binaryDescriptor[i];
            for (auto word : document.words)
                invertedIndex[word].push_back(document.id);
            imageDescriptors[document.id] = bs;
        }
        candidates.reserve(10000);
    }

    SearchResult search(const Document &document) const
    {
        bitset<256> bs;
        for (size_t i = 0; i < document.binaryDescriptor.size(); i++)
            bs[i] = document.binaryDescriptor[i];

        candidates.clear();
        pair<int, int> result = make_pair(1e5, -1);
        for (const auto &word : document.words)
        {
            auto it = invertedIndex.find(word);
            if (it == invertedIndex.end())
                continue;
            candidates.insert(candidates.end(), it->second.begin(), it->second.end());
        }

        for (const auto &it : candidates)
        {
            auto r = imageDescriptors.at(it);
            int distance = (bs ^ r).count();
            if (distance > result.first)
                continue;
            pair<int, int> tmpResult = make_pair(distance, it);
            if (tmpResult < result)
                result = tmpResult;
        }
        return SearchResult(result.second, result.first);
    }

private:
    mutable vector<int> candidates;
    unordered_map<int, vector<int>> invertedIndex;
    unordered_map<int, bitset<256>> imageDescriptors;
};




int main()
{
    const int DOCUMENT_NUMBER = 1e5;
    Corpus c1;

    for (int i = 0; i < DOCUMENT_NUMBER; i++)
    {
        Document doc = Document::generate_document(i + 1e8);
        c1.push_back(doc);
    }

    unique_ptr<Index> index1 = unique_ptr<Index>(new SimpleIndex(c1));
    unique_ptr<Index> index2 = unique_ptr<Index>(new OptimizedIndex(c1));

    vector<Document> testDocuments;
    for (int i = 0; i < 1000; i++)
        testDocuments.push_back(Document::generate_document(777, 30));

    vector<Index::SearchResult> searchResults1, searchResults2;

    auto start = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
    for (auto document : testDocuments)
        searchResults1.push_back(index1->search(document));
    float time1 = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count() - start;

    auto start2 = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
    for (auto document : testDocuments)
        searchResults2.push_back(index2->search(document));
    float time2 = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count() - start2;

    cout << "time 1: " << time1 << endl << "time 2: " << time2 << endl << "(time1 / time2) = " << time1 / time2 << endl;

    for(size_t i = 0; i < searchResults1.size(); i++)
    {
        if (searchResults1[i].documentID != searchResults2[i].documentID ||
                searchResults1[i].distance != searchResults2[i].distance)
        {
            cout << "not matched" << endl;
            cout << searchResults1[i].documentID << " " << searchResults1[i].distance << endl;
            cout << searchResults2[i].documentID << " " << searchResults2[i].distance << endl;
            return 0;
        }
    }
}

