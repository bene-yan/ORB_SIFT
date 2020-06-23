//
// Created by bene on 2020/5/25.
//

#include "../include/HomoDecomp.h"

#include "../Thirdparty/DBoW2/DUtils/Random.h"

namespace ORB_SIFT {

    HomoDecomp::HomoDecomp(cv::Mat &K,Frame &LastFrame,Frame &CurrentFrame,vector<int> &vnMatches12,int maxIterations)
    :mK(K),mLastFrame(LastFrame),mCurrentFrame(CurrentFrame),
     mvnMatches12(vnMatches12),mMaxIterations(maxIterations)
    {

    }

    void HomoDecomp::GenerateSets() {

        mPairMatch12.clear();
        mPairMatch12.reserve(mCurrentFrame.mvKeysUn.size());

        mvbMatched1.resize(mLastFrame.mvKeysUn.size());

        for (size_t i = 0, iend = mvnMatches12.size(); i < iend; i++) {
            if (mvnMatches12[i] >= 0) {
                mPairMatch12.push_back(make_pair(i, mvnMatches12[i]));
                mvbMatched1[i] = true;
            } else
                mvbMatched1[i] = false;
        }

        const int N = mPairMatch12.size();

        // Indices for minimum set selection
        vector <size_t> vAllIndices;
        vAllIndices.reserve(N);
        vector <size_t> vAvailableIndices;

        for (int i = 0; i < N; i++) {
            vAllIndices.push_back(i);
        }

        // Generate sets of 8 points for each RANSAC iteration
        mvSets = vector < vector < size_t > > (mMaxIterations, vector<size_t>(8, 0));

        DUtils::Random::SeedRandOnce(0);

        for (int it = 0; it < mMaxIterations; it++) {
            vAvailableIndices = vAllIndices;

            // Select a minimum set
            for (size_t j = 0; j < 8; j++)//TO-DO fix this for loop 匹配特征应该大于8
            {
                int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);

                int idx = vAvailableIndices[randi];

                mvSets[it][j] = idx;

                vAvailableIndices[randi] = vAvailableIndices.back();
                vAvailableIndices.pop_back();
            }
        }
    }

//TODO 如何在大量误匹配数据中找到正确的Homography
    void HomoDecomp::DecompHomography(float &score, cv::Mat &R21, cv::Mat &t21) {
        //步骤一：产生8点集
        GenerateSets();

        // Number of putative matches
        const int N = mPairMatch12.size();

        // Normalize coordinates
        //将mvkeys1和mvkeys2归一化到均值为0,一阶绝对矩为1,归一化矩阵分别为T1、T2
        vector <cv::Point2f> vPn1, vPn2;
        cv::Mat T1, T2;
        Normalize(mLastFrame.mvKeysUn, vPn1, T1);
        Normalize(mCurrentFrame.mvKeysUn, vPn2, T2);
        cv::Mat T2inv = T2.inv();

        // Best Results variables
        //score = 0.0;
        //vbMatchesInliers = vector<bool>(N, false);//通过对称重投影误差判断某个匹配是否为内点

        // Iteration variables
        vector <cv::Point2f> vPn1i(8);
        vector <cv::Point2f> vPn2i(8);
        cv::Mat H21i, H12i;
        //vector<bool> vbCurrentInliers(N, false);
        float currentScore;

        // Perform all RANSAC iterations and save the solution with highest score
        for (int it = 0; it < mMaxIterations; it++) {
            // Select a minimum set
            //每次迭代选8个点
            for (size_t j = 0; j < 8; j++) {
                int idx = mvSets[it][j];

                vPn1i[j] = vPn1[mPairMatch12[idx].first];
                vPn2i[j] = vPn2[mPairMatch12[idx].second];
            }

            //步骤二：计算H
            cv::Mat Hn = ComputeH21(vPn1i, vPn2i);   //无需修改
            H21i = T2inv * Hn * T1;
            H12i = H21i.inv();

            //cout << "H21" << it << ":" << endl << cv::format(H21i, cv::Formatter::FMT_C) << endl;

            //这里采用的方法是通过H矩阵计算匹配点的对称重投影误差
            //currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);

            //步骤三：从此次H中恢复R t n
            cv::Mat R21i, t21i;
            cv::Mat n1i;//(3,1,CV_32F);

            //TODO TO Debug
            if(false==ReconstructH(H21i, mK, R21i, t21i, n1i,vPn1i))//TODO 需要修改
                continue;   //未能恢复位姿

            //步骤四：三角化这个8个点，判断他们是否共面且平面法向量是否满足要求
            // Camera 1 Projection Matrix K[I|0]
            cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
            mK.copyTo(P1.rowRange(0, 3).colRange(0, 3));

            //cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);//相机1光心

            // Camera 2 Projection Matrix K[R|t]
            cv::Mat P2(3, 4, CV_32F);
            R21i.copyTo(P2.rowRange(0, 3).colRange(0, 3));
            t21i.copyTo(P2.rowRange(0, 3).col(3));
            P2 = mK * P2;

            //cv::Mat O2 = -R.t()*t;    //相机2光心

            cv::Mat p3dC1;
            vector <cv::Mat> vPn3dC1;

            for (size_t j = 0; j < 8; j++) {
                Triangulate(vPn1i[j], vPn2i[j], P1, P2, p3dC1);
                vPn3dC1.push_back(p3dC1);
            }


            cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
            if (true == isCoplanar(vPn3dC1, normal))
            {
                currentScore = normal.dot(n1i); //可能为负，最大为1
                //cout<<"eight point for H is coplanar,CurrentScore: "<<currentScore<<endl;
            }
            else
                //cout<<"eight point for H is non-coplanar"<<endl;


            if (currentScore > score) {
                R21 = R21i.clone();
                t21 = t21i.clone();
                //vbMatchesInliers = vbCurrentInliers;
                score = currentScore;
            }
        }


    }

    cv::Mat HomoDecomp::ComputeH21(const vector <cv::Point2f> &vP1, const vector <cv::Point2f> &vP2) {
        const int N = vP1.size();

        cv::Mat A(2 * N, 9, CV_32F);

        for (int i = 0; i < N; i++) {
            const float u1 = vP1[i].x;
            const float v1 = vP1[i].y;
            const float u2 = vP2[i].x;
            const float v2 = vP2[i].y;

            A.at<float>(2 * i, 0) = 0.0;
            A.at<float>(2 * i, 1) = 0.0;
            A.at<float>(2 * i, 2) = 0.0;
            A.at<float>(2 * i, 3) = -u1;
            A.at<float>(2 * i, 4) = -v1;
            A.at<float>(2 * i, 5) = -1;
            A.at<float>(2 * i, 6) = v2 * u1;
            A.at<float>(2 * i, 7) = v2 * v1;
            A.at<float>(2 * i, 8) = v2;

            A.at<float>(2 * i + 1, 0) = u1;
            A.at<float>(2 * i + 1, 1) = v1;
            A.at<float>(2 * i + 1, 2) = 1;
            A.at<float>(2 * i + 1, 3) = 0.0;
            A.at<float>(2 * i + 1, 4) = 0.0;
            A.at<float>(2 * i + 1, 5) = 0.0;
            A.at<float>(2 * i + 1, 6) = -u2 * u1;
            A.at<float>(2 * i + 1, 7) = -u2 * v1;
            A.at<float>(2 * i + 1, 8) = -u2;

        }

        cv::Mat u, w, vt;

        cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        return vt.row(8).reshape(0, 3);
    }

    void
    HomoDecomp::Normalize(const vector <cv::KeyPoint> &vKeys, vector <cv::Point2f> &vNormalizedPoints, cv::Mat &T) {
        float meanX = 0;
        float meanY = 0;
        const int N = vKeys.size();

        vNormalizedPoints.resize(N);

        for (int i = 0; i < N; i++) {
            meanX += vKeys[i].pt.x;
            meanY += vKeys[i].pt.y;
        }

        meanX = meanX / N;
        meanY = meanY / N;

        float meanDevX = 0;
        float meanDevY = 0;

        for (int i = 0; i < N; i++) {
            vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
            vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

            meanDevX += fabs(vNormalizedPoints[i].x);
            meanDevY += fabs(vNormalizedPoints[i].y);
        }

        meanDevX = meanDevX / N;
        meanDevY = meanDevY / N;

        float sX = 1.0 / meanDevX;
        float sY = 1.0 / meanDevY;

        for (int i = 0; i < N; i++) {
            vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
            vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
        }

        T = cv::Mat::eye(3, 3, CV_32F);
        T.at<float>(0, 0) = sX;
        T.at<float>(1, 1) = sY;
        T.at<float>(0, 2) = -meanX * sX;
        T.at<float>(1, 2) = -meanY * sY;
    }

    bool HomoDecomp::ReconstructH(cv::Mat &H21, cv::Mat &K, cv::Mat &R21, cv::Mat &t21, cv::Mat &n1,vector <cv::Point2f> vPn1i) {

        // We recover 8 motion hypotheses using the method of Faugeras et al.
        // Motion and structure from motion in a piecewise planar environment.
        // International Journal of Pattern Recognition and Artificial Intelligence, 1988

        cv::Mat invK = K.inv();
        cv::Mat A = invK * H21 * K;

        cv::Mat U, w, Vt, V;
        cv::SVD::compute(A, w, U, Vt, cv::SVD::FULL_UV);
        V = Vt.t();

        float s = cv::determinant(U) * cv::determinant(Vt);

        float d1 = w.at<float>(0);
        float d2 = w.at<float>(1);
        float d3 = w.at<float>(2);

        if (d1 / d2 < 1.00001 || d2 / d3 < 1.00001) {
            cout << "SVD abnormal!" << endl;
            return false;
        }

        vector <cv::Mat> vR, vt, vn;
        vector <float> vd;
        vR.reserve(8);
        vt.reserve(8);
        vn.reserve(8);
        vd.reserve(8);

        //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
        float aux1 = sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3));
        float aux3 = sqrt((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3));
        float x1[] = {aux1, aux1, -aux1, -aux1};    //x1,x3四种情况
        float x3[] = {aux3, -aux3, aux3, -aux3};

        //case d'=d2    //d'两种情况
        float aux_stheta = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2);

        float ctheta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
        float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

        for (int i = 0; i < 4; i++) {
            vd.push_back(s*d2);

            cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
            Rp.at<float>(0, 0) = ctheta;
            Rp.at<float>(0, 2) = -stheta[i];
            Rp.at<float>(2, 0) = stheta[i];
            Rp.at<float>(2, 2) = ctheta;

            cv::Mat R = s * U * Rp * Vt;
            vR.push_back(R);

            cv::Mat tp(3, 1, CV_32F);
            tp.at<float>(0) = x1[i];
            tp.at<float>(1) = 0;
            tp.at<float>(2) = -x3[i];
            tp *= d1 - d3;

            cv::Mat t = U * tp;
            //vt.push_back(t / cv::norm(t));//逻辑上，由于目的不同，这里不应该单位化，否则不知道真实尺度
            vt.push_back(t); //看看不单位化怎么样
            cv::Mat np(3, 1, CV_32F);
            np.at<float>(0) = x1[i];
            np.at<float>(1) = 0;
            np.at<float>(2) = x3[i];

            cv::Mat n = V * np;
            //if (n.at<float>(2) < 0)     //法向量做了处理，全都指向前方
                //n = -n;
            vn.push_back(n);
        }

        //case d'=-d2
        float aux_sphi = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2);

        float cphi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
        float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

        for (int i = 0; i < 4; i++) {
            vd.push_back(-s*d2);

            cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
            Rp.at<float>(0, 0) = cphi;
            Rp.at<float>(0, 2) = sphi[i];
            Rp.at<float>(1, 1) = -1;
            Rp.at<float>(2, 0) = sphi[i];
            Rp.at<float>(2, 2) = -cphi;

            cv::Mat R = s * U * Rp * Vt;
            vR.push_back(R);

            cv::Mat tp(3, 1, CV_32F);
            tp.at<float>(0) = x1[i];
            tp.at<float>(1) = 0;
            tp.at<float>(2) = x3[i];
            tp *= d1 + d3;

            cv::Mat t = U * tp;
            //vt.push_back(t / cv::norm(t));
            vt.push_back(t);

            cv::Mat np(3, 1, CV_32F);
            np.at<float>(0) = x1[i];
            np.at<float>(1) = 0;
            np.at<float>(2) = x3[i];

            cv::Mat n = V * np;
            //if (n.at<float>(2) < 0)
                //n = -n;
            vn.push_back(n);
        }

        //以上代码恢复了8个解

        int bestGood = 0;
        int secondBestGood = 0;
        int bestSolutionIdx = -1;
        float bestParallax = -1;
        vector <cv::Point3f> bestP3D;
        vector<bool> bestTriangulated;

        // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
        // We reconstruct all hypotheses and check in terms of triangulated points and parallax
        //从8个解中剔除错误解
        for (size_t i = 0; i < 8; i++) {
            //cout << "----Solution[" << i << "]----" << endl;
            //float parallaxi;
            //vector <cv::Point3f> vP3Di;
            //vector<bool> vbTriangulatedi;
/*
            //条件一：(剩4组解)
            //Both frames,F* and F must be in the same side of the object plane.
            //1+n.t()*R.t()*t>0
            //cv::Mat nt=vn[i].t();
            cv::Mat Rt = vR[i].t();
            cv::Mat t = vt[i]/vd[i];
            double d = vn[i].dot(Rt * t)+1.0;
            //double d=vn[i].t()*vR[i].t()*vt[i]+1;
            cout<<"d"<<i<<": "<<d<<endl;
            if (d <= 0.000001)
                continue;

*/

            if(false==CheckVisibility(H21,vPn1i,vd[i]))
                continue;


            //条件二：(剩两组解)
            //For all the reference points being visible, they must be in front of the camera.
            //m*.t()n*>0;==//m.t()(Rn)>0;
            //条件二From Faguras
            //ntm1/d>0
            int nGood = CheckRT(mPairMatch12, vR[i], vn[i],vd[i]);
            //cout << "nGood[" << i << "]:" << nGood <<"/"<<mPairMatch12.size()<< endl;
            //int nGood = CheckRT(vR[i],vt[i],mLastFrame.mvKeysUn,mCurrentFrame.mvKeysUn,mPairMatch12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);

            if (nGood < 0.5 * mPairMatch12.size())
                continue;

            //TODO 需要第三个条件选出最终解


            //能活到这里的解只剩下两个。
            cv::Mat t_gt;
            t_gt=vt[i]/vd[i]*1.65;
            //cout << "t_gt" << i << ":" << endl << cv::format(t_gt, cv::Formatter::FMT_C) << endl;

            cv::Mat t0(3,1,CV_32F);
            t0.at<float>(0)=-0.04690294;
            t0.at<float>(1)=-0.02839928;
            t0.at<float>(2)=0.8586941;

            if(cv::norm(t_gt-t0)<0.05||cv::norm(t_gt+t0)<0.05)
            {
                //cout << "R" << i << ":" << endl << cv::format(vR[i], cv::Formatter::FMT_C) << endl;
                //cout << "t" << i << ":" << endl << cv::format(vt[i], cv::Formatter::FMT_C) << endl;
            }

            //cout << "R" << i << ":" << endl << cv::format(vR[i], cv::Formatter::FMT_C) << endl;
            //cout << "t" << i << ":" << endl << cv::format(vt[i], cv::Formatter::FMT_C) << endl;
            //cout << "nGood " << nGood << "v.s." << " bestGood" << bestGood << endl;


            if (nGood > bestGood) {
                //secondBestGood = bestGood;
                bestGood = nGood;
                bestSolutionIdx = i;
                //bestParallax = parallaxi;
                //bestP3D = vP3Di;
                //bestTriangulated = vbTriangulatedi;
            }
            /*
            else if(nGood>secondBestGood)
            {
                secondBestGood = nGood;
            }*/

        }

        //if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
        //if(bestGood>0.9*N)
        if(bestSolutionIdx>=0)
        {
            //TO-DO bug here
            //Debug: bestSolutionIdx could = -1!

            vR[bestSolutionIdx].copyTo(R21);
            vt[bestSolutionIdx].copyTo(t21);
            vn[bestSolutionIdx].copyTo(n1);
            //vP3D = bestP3D;
            //vbTriangulated = bestTriangulated;

            return true;
        }
        cout << "get to end, so false" << endl;
        return false;
    }
/*
    int HomoDecomp::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector <cv::KeyPoint> &vKeys1,
                            const vector <cv::KeyPoint> &vKeys2,
                            const vector <Match> &vMatches12,
                            const cv::Mat &K, vector <cv::Point3f> &vP3D, float th2, vector<bool> &vbGood,
                            float &parallax) {
        // Calibration parameters
        const float fx = K.at<float>(0, 0);
        const float fy = K.at<float>(1, 1);
        const float cx = K.at<float>(0, 2);
        const float cy = K.at<float>(1, 2);

        vbGood = vector<bool>(vKeys1.size(), false);
        vP3D.resize(vKeys1.size());

        vector<float> vCosParallax;
        vCosParallax.reserve(vKeys1.size());

        // Camera 1 Projection Matrix K[I|0]
        cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
        K.copyTo(P1.rowRange(0, 3).colRange(0, 3));

        cv::Mat O1 = cv::Mat::zeros(3, 1, CV_32F);

        // Camera 2 Projection Matrix K[R|t]
        cv::Mat P2(3, 4, CV_32F);
        R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
        t.copyTo(P2.rowRange(0, 3).col(3));
        P2 = K * P2;

        cv::Mat O2 = -R.t() * t;

        int nGood = 0;

        for (size_t i = 0, iend = vMatches12.size(); i < iend; i++) {
            //if(!vbMatchesInliers[i])
            //continue;

            const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
            const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
            cv::Mat p3dC1;

            Triangulate(kp1, kp2, P1, P2, p3dC1);

            if (!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2))) {
                vbGood[vMatches12[i].first] = false;
                continue;
            }

            // Check parallax
            cv::Mat normal1 = p3dC1 - O1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = p3dC1 - O2;
            float dist2 = cv::norm(normal2);

            float cosParallax = normal1.dot(normal2) / (dist1 * dist2);

            // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
            if (p3dC1.at<float>(2) <= 0 && cosParallax < 0.99998)
                continue;

            // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
            cv::Mat p3dC2 = R * p3dC1 + t;

            if (p3dC2.at<float>(2) <= 0 && cosParallax < 0.99998)
                continue;

            // Check reprojection error in first image
            float im1x, im1y;
            float invZ1 = 1.0 / p3dC1.at<float>(2);
            im1x = fx * p3dC1.at<float>(0) * invZ1 + cx;
            im1y = fy * p3dC1.at<float>(1) * invZ1 + cy;

            float squareError1 = (im1x - kp1.pt.x) * (im1x - kp1.pt.x) + (im1y - kp1.pt.y) * (im1y - kp1.pt.y);

            if (squareError1 > th2)
                continue;

            // Check reprojection error in second image
            float im2x, im2y;
            float invZ2 = 1.0 / p3dC2.at<float>(2);
            im2x = fx * p3dC2.at<float>(0) * invZ2 + cx;
            im2y = fy * p3dC2.at<float>(1) * invZ2 + cy;

            float squareError2 = (im2x - kp2.pt.x) * (im2x - kp2.pt.x) + (im2y - kp2.pt.y) * (im2y - kp2.pt.y);

            if (squareError2 > th2)
                continue;

            //总是有其中两个解得到相同数量的nGood
            //增加约束Both frames,F* and F must be in the same side of the object plane.
            //1+n.t()*R.t()*t>0

            vCosParallax.push_back(cosParallax);
            vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
            nGood++;

            if (cosParallax < 0.99998)
                vbGood[vMatches12[i].first] = true;
        }

        if (nGood > 0) {
            sort(vCosParallax.begin(), vCosParallax.end());

            size_t idx = min(50, int(vCosParallax.size() - 1));
            parallax = acos(vCosParallax[idx]) * 180 / CV_PI;
        } else
            parallax = 0;

        return nGood;
    }
*/
    int HomoDecomp::CheckRT(vector <Match> vMatches12, cv::Mat R, cv::Mat N,float d) {
        int nGood = 0;
        for (size_t i = 0, iend = vMatches12.size(); i < iend; i++) {
            //条件二：
            //For all the reference points being visible, they must be in front of the camera.
            //m*.t()n*>0;==m.t()(Rn)>0;

            cv::Point pt1 = mLastFrame.mvKeysUn[mPairMatch12[i].first].pt;
            cv::Point pt2 = mCurrentFrame.mvKeysUn[mPairMatch12[i].second].pt;
            cv::Mat m(3, 1, CV_32F);
            cv::Mat m_s(3, 1, CV_32F);
            m.at<float>(0) = pt1.x;
            m.at<float>(1) = pt1.y;
            m.at<float>(2) = 1.0;

            m_s.at<float>(0) = pt2.x;
            m_s.at<float>(1) = pt2.y;
            m_s.at<float>(2) = 1.0;

            //if (m.dot(R * N) > 0.000001 && m_s.dot(N) > 0.000001)
                //nGood++;

            //条件二From Faguras
            //ntm1/d>0
            if(N.dot(m)/d>0.000001)
                nGood++;
        }

        return nGood;
    }

    bool HomoDecomp::CheckVisibility(cv::Mat &H21,vector<cv::Point2f> vPn1i,float di){
        float a31=H21.at<float>(2,0);
        float a32=H21.at<float>(2,1);
        float a33=H21.at<float>(2,2);
        for(size_t i=0,iend=vPn1i.size();i<iend;i++)
        {
            // Z2    a31*x1+a32*y1+a33
            // -- ==------------------- >0
            // Z1           d
            float Z2divZ1=(a31*vPn1i[i].x+a32*vPn1i[i].y+a33)/di;
            if(Z2divZ1<0)
                return false;
        }

        return true;

    }

    void HomoDecomp::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2,
                                 cv::Mat &x3D) {
        cv::Mat A(4, 4, CV_32F);

        A.row(0) = kp1.pt.x * P1.row(2) - P1.row(0);
        A.row(1) = kp1.pt.y * P1.row(2) - P1.row(1);
        A.row(2) = kp2.pt.x * P2.row(2) - P2.row(0);
        A.row(3) = kp2.pt.y * P2.row(2) - P2.row(1);

        cv::Mat u, w, vt;
        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        x3D = vt.row(3).t();
        x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
    }

    void HomoDecomp::Triangulate(const cv::Point &pt1, const cv::Point &pt2, const cv::Mat &P1, const cv::Mat &P2,
                                 cv::Mat &x3D) {
        cv::Mat A(4, 4, CV_32F);

        A.row(0) = pt1.x * P1.row(2) - P1.row(0);
        A.row(1) = pt1.y * P1.row(2) - P1.row(1);
        A.row(2) = pt2.x * P2.row(2) - P2.row(0);
        A.row(3) = pt2.y * P2.row(2) - P2.row(1);

        cv::Mat u, w, vt;
        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        x3D = vt.row(3).t();
        x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
    }

    //TODO 考虑拟合平面而不是严格共面
    bool HomoDecomp::isCoplanar(vector <cv::Mat> &Points3D, cv::Mat &normal) {

        cv::Mat P1 = Points3D[0];
        cv::Mat P2 = Points3D[1];
        cv::Mat L12 = P2 - P1;
        cv::Mat P3;
        for(size_t j=2,N = Points3D.size();j<N;j++)
        {
            P3=Points3D[j];
            if(L12.dot(P3)>0||L12.dot(P3)<0)//如果不共线
                break;
        }
        if(L12.dot(P3)==0)//搜索结束依然共线
            return false;

        cv::Mat L13 = P3 - P1;
        normal = L12.cross(L13);    //共线重新选点
        normal = normal/cv::norm(normal);
        if(normal.at<float>(2)<0)
            normal=-normal;

        cv::Mat Pi(3,1,CV_32F);
        for (size_t i = 2, N = Points3D.size(); i < N; Pi = Points3D[i++]) {
            if (Pi.dot(normal) > 0.001) {
                normal = cv::Mat::zeros(3, 1, CV_32F);
                return false;
            }
        }

        return true;
    }

    void HomoDecomp::eliminateWrongMatch() {
        size_t N=mPairMatch12.size();


    }
}
