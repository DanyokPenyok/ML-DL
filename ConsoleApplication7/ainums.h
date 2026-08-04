#pragma once
#include <cmath>
#include <vector>
#include <iostream>
#include <random>

using namespace std;

class ainums
{
private:
	double bias = 0.0;
	vector<double> w;
	vector<vector <double>> x;
	vector <double> xtrue;
	vector <double> probability;
	int epoches;
	int xsize; //размер подмассива
	int sizearray; //размер массива
	double lr;

public:
	ainums(const vector<vector<double>>& xinit, double learningrate,int epoch) {
		lr = learningrate;
		epoches = epoch;
		sizearray = xinit.size();
		xsize = xinit[0].size() - 1;
		w.resize(xsize);
		x.resize(sizearray, vector<double>(xsize));
		xtrue.resize(sizearray);
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<double> dist(-0.5, 0.5);
		for (size_t i = 0; i < xsize; i++) {
			w[i] = dist(gen);
		}
		for (int i = 0; i < sizearray; i++)
		{
			for (int j = 0; j < xsize; j++)
			{
				x[i][j] = xinit[i][j];
			}
			xtrue[i] = xinit[i][xsize];
		}
	}
	void addnums(const vector <double>&nums) {
		if (nums.size() != xsize + 1) {
			cout << "addnums error" << endl;
			return;
		}
		vector<double>nums2x(nums.begin(), nums.end() - 1);
		x.push_back(nums2x);
		xtrue.push_back(nums.back());
		sizearray = x.size();
	}

	void z() {
		probability.resize(sizearray);
		for (int i = 0; i < sizearray; i++)
		{
			double z = 0;
			for (int j = 0; j < xsize; j++)
			{
				z += x[i][j] * w[j];
			}
			z += bias;
			probability[i] = (1 / (1 + exp(-z)));
		}
	}
	void learning(){
		for (int i = 0; i < epoches; i++)
		{
			z();
			for (int j = 0; j < sizearray; j++)
			{
				double error = (probability[j] - xtrue[j]) * probability[j] * (1 - probability[j]);
				for (int k = 0; k < xsize; k++)
				{
					w[k] = w[k] - (lr * error * x[j][k]);
					
				}
				bias = bias - (lr * error);
			}
		}
	}

	
	void print() {
		for (int i = 0; i < w.size(); i++)
		{
			cout << w[i] << endl;
			
		}
		z();
		for (int i = 0; i < sizearray; i++) {
			cout << "\ncтрока " << i << "\nожидание " << xtrue[i]
				<< "\nответ: " << probability[i] << endl;
		}

	}
};
