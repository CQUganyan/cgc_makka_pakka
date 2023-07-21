#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cblas.h>  //CBLAS 是一组用于执行基本线性代数运算（如矩阵乘法、向量加法等）的标准函数接口
                    //它提供了高性能的数学函数，特别是针对线性代数运算的优化，如矩阵乘法、向量加法、矩阵求逆等

using namespace std;

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;
int e_num = 0;
int F0 = 0, F1 = 0, F2 = 0;

vector<int> row_ptr;
vector<int> col_index;
vector<float> edge_val;

vector<int> degree;
vector<int> raw_graph;

float *X0, *W1, *W2, *X1, *X2, *X_inter;


void readGraph(char *fname) //读取图结构数据
{
    ifstream infile(fname);

    int source;
    int end;

    infile >> v_num >> e_num;

    while (!infile.eof())
    {
        infile >> source >> end;
        if (infile.peek() == EOF)
            break;
        raw_graph.push_back(source);
        raw_graph.push_back(end);
    }
}


void raw_graph_to_CSR_single() //将原始图数据raw_graph转成CSR格式(单线程)
{
    int src;
    int dst;
    row_ptr.resize(v_num + 1, 0);
    degree.resize(v_num, 0);

    for (int i = 0; i < raw_graph.size() / 2; i++)
    {
        src = raw_graph[2 * i];
        degree[src]++;
    }
    for (int i = 1; i <= v_num; i++)
    {
        row_ptr[i] = row_ptr[i - 1] + degree[i - 1];  //行偏移数组,存储每行的起始索引在值数组和列索引数组中的位置
    }

    col_index.resize(e_num);
    edge_val.resize(e_num, 0.0);

    vector<int> curr_pos(v_num, 0);
    for (int i = 0; i < raw_graph.size() / 2; i++)
    {
        src = raw_graph[2 * i];
        dst = raw_graph[2 * i + 1];
        int index = row_ptr[src] + curr_pos[src];
        col_index[index] = dst;                   //列索引数组,存储每个非零元素所在的列索引
        edge_val[index] = 1 / sqrt(degree[src]) / sqrt(degree[dst]);   //归一化操作;  
        curr_pos[src]++;
    }
}

/*
矩阵乘AX和激活函数ReLU融合(单线程)
*/
void AX_relu_single(int dim, float *in_X, float *out_X)
{
    for (int i = 0; i < v_num; i++)
    {
        int start = row_ptr[i];
        int end = row_ptr[i + 1];
        float sum[dim] = {0};

        for (int j = start; j < end; j++)
        {
            int nbr = col_index[j];
            float val = edge_val[j];
            for (int k = 0; k < dim; k++)   //将每个节点的特征向量内的元素聚合分配给所有核
            {
                sum[k] += val * in_X[nbr * dim + k];
            }
        }
        memcpy(&out_X[i * dim], sum, dim * sizeof(float));

        for(int k1 = 0; k1 < dim; k1++){                    //Relu激活函数
            if(out_X[i * dim + k1]<0) out_X[i * dim + k1]=0;
        }

    }
}

/*
矩阵乘AX(单线程)
*/
void AX_single(int dim, float *in_X, float *out_X)  
{
    for (int i = 0; i < v_num; i++)
    {
        int start = row_ptr[i];
        int end = row_ptr[i + 1];
        float sum[dim] = {0};

        for (int j = start; j < end; j++)
        {
            int nbr = col_index[j];
            float val = edge_val[j];
            for (int k = 0; k < dim; k++)   //将每个节点的特征向量内的元素聚合分配给所有核
            {
                sum[k] += val * in_X[nbr * dim + k];
            }
        }
        memcpy(&out_X[i * dim], sum, dim * sizeof(float));

    }
}
/*激活函数LogSoftmax(单线程)*/
void LogSoftmax_single(int dim, float *X)            
{
    for (int i = 0; i < v_num; i++)
    {
        float max_val = X[i * dim];    
        for (int j = 1; j < dim; j++)
        {
            if (X[i * dim + j] > max_val)
                max_val = X[i * dim + j];       //使用一维数组来模拟二维数组
        }

        float sum_exp = 0;
        for (int j = 0; j < dim; j++)
        {
            sum_exp += expf(X[i * dim + j] - max_val);
        }

        float log_sum_exp = logf(sum_exp);

        for (int j = 0; j < dim; j++)
        {
            X[i * dim + j] = X[i * dim + j] - max_val - log_sum_exp;   //使用一维数组来模拟二维数组
        }
    }
}

void raw_graph_to_CSR()  //将原始图数据raw_graph转成CSR格式,使用omp并行
{
    int src;
    int dst;

    row_ptr.resize(v_num + 1, 0);
    degree.resize(v_num, 0);

    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < raw_graph.size() / 2; i++)
    {
        src = raw_graph[2 * i];
        //原子操作
        #pragma omp atomic
        degree[src]++;
    }
    for (i = 1; i <= v_num; i++)
    {
        row_ptr[i] = row_ptr[i - 1] + degree[i - 1]; //行偏移数组,存储每行的起始索引在值数组和列索引数组中的位置
    }

    col_index.resize(e_num);
    edge_val.resize(e_num, 0.0);

    vector<int> curr_pos(v_num, 0);
    int index;
    #pragma omp parallel for private(i,src,dst,index)
    for (i = 0; i < raw_graph.size() / 2; i++)
    {
        src = raw_graph[2 * i];
        dst = raw_graph[2 * i + 1];
        index = row_ptr[src] + curr_pos[src];
        col_index[index] = dst;                                //列索引数组,存储每个非零元素所在的列索引
        edge_val[index] = 1 / sqrt(degree[src]) / sqrt(degree[dst]);  //归一化操作
        #pragma omp atomic       
        curr_pos[src]++;
    }
}

void readFloat(char *fname, float *&dst, int num)
{
    dst = (float *)malloc(num * sizeof(float));
    FILE *fp = fopen(fname, "rb");
    fread(dst, num * sizeof(float), 1, fp);
    fclose(fp);
}

void initFloat(float *&dst, int num)
{
    dst = (float *)malloc(num * sizeof(float));
    memset(dst, 0, num * sizeof(float));
}

/*
密集矩阵相乘
调用openblas库的矩阵乘函数进行优化
*/
void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, v_num, out_dim, in_dim, 1.0, in_X, in_dim, W, out_dim, 0.0, out_X, out_dim);
}

/*
矩阵乘AX和激活函数ReLU融合(多线程)
适用于图数据大于500K的数据集
*/
void AX_relu(int dim, float *in_X, float *out_X)
{
    int start,end;
    int nbr;
    float val;
    int i, j, k, k1;
    #pragma omp parallel for private(i, j, k, k1, start, end, nbr, val)
    for (i = 0; i < v_num; i++)
    {
        start = row_ptr[i];
        end = row_ptr[i + 1];
        float sum[dim] = {0};

        for (j = start; j < end; j++)
        {
            nbr = col_index[j];
            val = edge_val[j];
            for (k = 0; k < dim; k++)    //将每个节点的特征向量内的元素聚合分配给所有核
            {
                sum[k] += val * in_X[nbr * dim + k];
            }
        }
        memcpy(&out_X[i * dim], sum, dim * sizeof(float));
        for(k1 = 0;k1 < dim;k1++){
            if(out_X[i * dim + k1]<0) out_X[i * dim + k1]=0;             //使用一维数组来模拟二维数组
        }

    }
}

/*
矩阵乘AX和激活函数LogSoftmax融合(多线程)
适用于图数据大于500K的数据集
*/
void AX_LogSoftmax(int dim, float *in_X, float *out_X)
{
    #pragma omp parallel for
    for (int i = 0; i < v_num; i++)
    {
        int start = row_ptr[i];
        int end = row_ptr[i + 1];
        float sum[dim] = {0};

        for (int j = start; j < end; j++)
        {
            int nbr = col_index[j];
            float val = edge_val[j];
            #pragma omp simd
            for (int k = 0; k < dim; k++)
            {
            sum[k] += val * in_X[nbr * dim + k];
            }
        }

        float max_val = sum[0];
        for (int j = 1; j < dim; j++)
        {
            if (sum[j] > max_val)
            max_val = sum[j];
        }

        float sum_exp = 0;
        // #pragma omp simd reduction(+:sum_exp)
        for (int j = 0; j < dim; j++)
        {
        sum_exp += expf(sum[j] - max_val);
        }

        float log_sum_exp = logf(sum_exp);

        for (int j = 0; j < dim; j++)
        {
        out_X[i * dim + j] = sum[j] - max_val - log_sum_exp;
        }
    }
}

float MaxRowSum(float *X, int dim)
{
    float max_sum = -__FLT_MAX__;
    // #pragma omp parallel for reduction(max:max_sum)
    for (int i = 0; i < v_num; i++)
    {
        float sum = 0;
        for (int j = 0; j < dim; j++)
        {
            sum += X[i * dim + j];
        }
        
        if (sum > max_sum)
            max_sum = sum;
    }
    return max_sum;
}

void freeFloats()
{
    free(X0);
    free(W1);
    free(W2);
    free(X1);
    free(X2);
    free(X_inter);
}

void cleanup()
{
    degree.clear();
    row_ptr.clear();
    col_index.clear();
    edge_val.clear();
    freeFloats();
}

int main(int argc, char **argv)
{
    F0 = atoi(argv[1]);
    F1 = atoi(argv[2]);
    F2 = atoi(argv[3]);

    readGraph(argv[4]);
    readFloat(argv[5], X0, v_num * F0);
    readFloat(argv[6], W1, F0 * F1);
    readFloat(argv[7], W2, F1 * F2);

    initFloat(X1, v_num * F1);
    initFloat(X_inter, v_num * max(F1,F2));
    initFloat(X2, v_num * F2);

    TimePoint start = chrono::steady_clock::now();

    openblas_set_num_threads(1);

    if(v_num>=500000){                        //根据节点数做出判断，当图数据大于500K时，启用openMP多线程指令
        raw_graph_to_CSR();

        XW(F0, F1, X0, X_inter, W1);

        AX_relu(F1, X_inter, X1);

        XW(F1, F2, X1, X_inter, W2);

        AX_LogSoftmax(F2, X_inter, X2);
    }
    else{
        raw_graph_to_CSR_single();

        XW(F0, F1, X0, X_inter, W1);

        AX_relu_single(F1, X_inter, X1);

        XW(F1, F2, X1, X_inter, W2);

        AX_single(F2, X_inter, X2);

        LogSoftmax_single(F2, X2);
    }
    float max_sum = MaxRowSum(X2, F2);

    TimePoint end = chrono::steady_clock::now();
    chrono::duration<double> l_durationSec = end - start;
    double l_timeMs = l_durationSec.count() * 1e3;

    printf("%.8f\n", max_sum);
    printf("%.8lf\n", l_timeMs);

    // ofstream fout("ubuntu_optimized3.txt", ios::app);
    // ofstream fout("ubuntu_optimized3.txt", ios::app);
    // fout << l_timeMs << endl;

    cleanup();

    return 0;
}
