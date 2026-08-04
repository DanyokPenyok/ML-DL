#pragma once
#include <cmath>
#include <vector>
#include <iostream>
#include <random>
using namespace std;
class ainumsdeep
{
private:
	int initlength; //колво комбинаций
	int initcombolength; //колво цифер в комбинации
	int hiddenneuronsC;
	int outneuronsCount;
	int epoches;
	double alpha;
	vector<vector<double>> X; //исходные данные
	vector<double> xtrue; // отвееты
	vector<vector<double>> whide;// веса из x12... в H12...
	vector<vector<double>> wout; // из H12... в out
	vector<double>out;
	vector<double> H; //Z
	vector<double> biasHide;
	vector<double> biasOut;



public:
	ainumsdeep(const vector<vector<double>>& initX, int epoch, double a, int hiddenneuronsCC, int outneuronsC) {
		initlength = initX.size();
		epoches = epoch;
		alpha = a;
		hiddenneuronsC = hiddenneuronsCC;
		outneuronsCount = outneuronsC;
		initcombolength = initX[0].size() - 1;
		X.resize(initlength, vector<double>(initcombolength));
		xtrue.resize(initlength);
		biasHide.resize(hiddenneuronsC);
		biasOut.resize(outneuronsC);
		whide.resize(initcombolength, vector<double>(hiddenneuronsC));
		wout.resize(outneuronsC, vector<double>(hiddenneuronsC));
		out.resize(outneuronsC);
		H.resize(hiddenneuronsC);
		for (size_t i = 0; i < initlength; i++)
		{
			for (size_t j = 0; j < initcombolength; j++)
			{
				X[i][j] = initX[i][j];
			}
			xtrue[i] = initX[i][initcombolength];
		}
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<double> dist(-0.5, 0.5);
		for (size_t i = 0; i < initcombolength; i++)
		{
			for (size_t j = 0; j < hiddenneuronsC; j++)
			{
				whide[i][j] = dist(gen);
			}
		}
		for (size_t i = 0; i < outneuronsC; i++)
		{
			for (size_t j = 0; j < hiddenneuronsC; j++)
			{
				wout[i][j] = dist(gen);
			}
		}
		for (size_t i = 0; i < biasHide.size(); i++)
		{
			biasHide[i] = dist(gen);
		}
		for (size_t i = 0; i < biasOut.size(); i++)
		{
			biasOut[i] = dist(gen);
		}
	}

	void forward(int rowindex) {
		//Z H
		for (size_t i = 0; i < hiddenneuronsC; i++)
		{
			double Z = 0;
			for (size_t j = 0; j < initcombolength; j++)
			{
				Z += whide[j][i] * X[rowindex][j];
			}
			Z += biasHide[i];
			H[i] = 1.0 / (1.0 + exp(-Z));
		}
		for (size_t i = 0; i < outneuronsCount; i++)
		{
			double sum = 0;
			for (size_t j = 0; j < hiddenneuronsC; j++)
			{
				sum += H[j] * wout[i][j];
			}
			sum += biasOut[i];
			out[i] = 1.0 / (1.0 + exp(-sum));
		}
	}
	void backward(int rowindex) {
		//ошибка out + delta
		vector<double> deltaout;
		deltaout.resize(outneuronsCount);
		for (size_t i = 0; i < outneuronsCount; i++)
		{
			double error = xtrue[rowindex] - out[i];
			deltaout[i] = error * out[i] * (1 - out[i]);
		}
		vector<double> deltaH(hiddenneuronsC);
		for (size_t i = 0; i < deltaH.size(); i++)
		{
			double sum = 0;
			for (size_t j = 0; j < deltaout.size(); j++)
			{
				sum += deltaout[j] * wout[j][i];
			}
			deltaH[i] = sum * H[i] * (1 - H[i]);
			biasHide[i] = biasHide[i] + (deltaH[i] * alpha);
		}
		for (size_t i = 0; i < outneuronsCount; i++)
		{
			for (size_t j = 0; j < hiddenneuronsC; j++)
			{
				wout[i][j] = wout[i][j] + (alpha * deltaout[i] * H[j]);
			}
			biasOut[i] = biasOut[i] + (alpha * deltaout[i]);
		} 
		for (size_t i = 0; i < whide.size(); i++)
		{
			for (size_t j = 0; j < whide[0].size(); j++)
			{
				whide[i][j] = whide[i][j] + (alpha * deltaH[j] * X[rowindex][i]);
			}
		}
	}

	void learning() {
		for (size_t i = 0; i < epoches; i++)
		{
			for (size_t j = 0; j < initlength; j++)
			{
				forward(j);
				backward(j);
			}
		}
	}
	void print() {
		for (size_t i = 0; i < initlength; i++)
		{
			cout << "elementarray "<<i + 1 << " xtrue: " << xtrue[i] << " return(out[i]): ";
			forward(i);
			for (size_t j = 0; j < out.size(); j++)
			{
				cout << out[j] << endl;
			}
		}
	}
};

