
#include <NSS.h>
#include <glm/gtx/norm.hpp>
#include <glm/glm.hpp>
#include <iostream>
#include <ppl.h>

void NSS::normalSpaceSampling(const float &sample_rate, std::vector<glm::fvec3> &sampled_vertices, std::vector<glm::fvec3> &sampled_normals)
{
	std::vector< std::vector<int> > normbuckets;
	sort_into_buckets(m_original_normals, normbuckets);
	int ndesired = int(ceil(sample_rate * m_original_vertices.size()));
	sampled_vertices.clear();  sampled_normals.clear();
	while (sampled_vertices.size() < ndesired)
	{
		for (int i = 0; i < normbuckets.size(); i++)
		{
			if (!normbuckets[i].empty())
			{
				int ind = normbuckets[i].back();
				sampled_vertices.emplace_back(m_original_vertices[ind]);
				sampled_normals.emplace_back(m_original_normals[ind]);
				normbuckets[i].pop_back();
			}
		}
	}
}

void NSS::sort_into_buckets(const std::vector<glm::fvec3> &n, std::vector< std::vector<int>> &normbuckets)
{
	const int Q = 4;
	const float Qsqrt1_2 = 2.8284f;
	normbuckets.resize(3 * Q*Q);
	for (int i = 0; i < n.size(); i++)
	{
		const float *N = &n[i][0];

		float ax = fabs(N[0]), ay = fabs(N[1]), az = fabs(N[2]);
		int A;
		float u, v;
		if (ax > ay)
		{
			if (ax > az)
			{
				A = 0;
				u = (N[0] > 0) ? N[1] : -N[1];
				v = (N[0] > 0) ? N[2] : -N[2];
			}
			else {
				A = 2;
				u = (N[2] > 0) ? N[0] : -N[0];
				v = (N[2] > 0) ? N[1] : -N[1];
			}
		}
		else
		{
			if (ay > az)
			{
				A = 1;
				u = (N[1] > 0) ? N[2] : -N[2];
				v = (N[1] > 0) ? N[0] : -N[0];
			}
			else
			{
				A = 2;
				u = (N[2] > 0) ? N[0] : -N[0];
				v = (N[2] > 0) ? N[1] : -N[1];
			}
		}
		int U = int(u * Qsqrt1_2) + (Q / 2);
		int V = int(v * Qsqrt1_2) + (Q / 2);
		normbuckets[((A * Q) + U) * Q + V].push_back(i);
	}
	for (int bucket = 0; bucket < normbuckets.size(); bucket++)
		std::random_shuffle(normbuckets[bucket].begin(), normbuckets[bucket].end());
}

void DNSS::dualNormalSpaceSampling(const float &sample_rate, std::vector<glm::fvec3> &sampled_vertices, std::vector<glm::fvec3> &sampled_normals)
{

	//Timer timer;
//	timer.start();
	computeCentroidandNormalize();
	computeRotationalNormals();
	computeRotationalReturn();

	sortIntoBucket();
	initBucketB();
	int ndesired = int(ceil(sample_rate * m_vertices_original.size()));

	sampled_vertices.clear();  sampled_normals.clear();
	sampled_vertices.reserve(ndesired);  sampled_normals.reserve(ndesired);

	while (sampled_vertices.size() < ndesired)
	{
		int pid = pickPoint();

		sampled_vertices.emplace_back(m_vertices_original[pid]);
		sampled_normals.emplace_back(m_normals_original[pid]);
		//updateBucketOrder();
	}
	sampled_vertices.shrink_to_fit();  sampled_normals.shrink_to_fit();
	//	timer.stop();
	//	std::wcout << L"DNSS sampling time : " << timer.peek_msec() << L" (ms)" << std::endl;
}

void DNSS::computeRotationalReturn()
{
	int nSize = static_cast<int>(m_vertices_normalized.size());
	m_rotationalReturns.clear();
	m_rotationalReturns.resize(nSize);

	Concurrency::parallel_for(0, nSize, [&](int idx)
	{
		float dot_pn = glm::dot(glm::normalize(m_vertices_normalized[idx]), m_normals_original[idx]);
		float beta = acos(dot_pn);// / (glm::l2Norm(m_vertices_normalized[idx])*glm::l2Norm(m_normals_original[idx])));
		float p_abs = glm::l2Norm(m_vertices_normalized[idx]);
		float pp_abs = 2 * p_abs *sin(m_thetaForReturn / 2);

		float pq_abs_positive = pp_abs * cos(beta - m_thetaForReturn / 2);
		float pq_abs_negative = pp_abs * cos(-beta - m_thetaForReturn / 2);
		float sin_beta = sin(beta);
		float cos_beta = cos(beta);

		float atanbeta_positive = atan((pq_abs_positive*sin_beta) / (p_abs - pq_abs_positive * cos_beta));
		float gamma_positive = m_thetaForReturn - atanbeta_positive;

		float atanbeta_negative = atan((pq_abs_negative*(-sin_beta)) / (p_abs - pq_abs_negative * cos_beta));
		float gamma_negative = m_thetaForReturn - atanbeta_negative;

		float rotationalReturnPositive = p_abs * gamma_positive / m_thetaForReturn;
		float rotationalReturnNegative = p_abs * gamma_negative / m_thetaForReturn;

		float RR = fmax(rotationalReturnPositive, rotationalReturnNegative);
		m_rotationalReturns[idx] = RR;
	}
	);
}
void DNSS::initBucketB()
{
	m_bucketList_T.clear();
	m_bucketList_R.clear();
	m_bucketList_T.resize(m_bucketTranslation.size());
	m_bucketList_R.resize(m_bucketRotation.size());
	const int bucketsize_R = static_cast<int>(m_bucketRotation.size());
	const int bucketsize_T = static_cast<int>(m_bucketTranslation.size());
	for (int idx = 0; idx < bucketsize_R; idx++)
	{
		m_bucketList_R[idx].bucketIndex = idx;
		//	m_bucketRotation[idx].clear();
		if (m_bucketRotation[idx].empty())
			m_bucketList_R[idx].constraints = FLT_MAX;
		else
			m_bucketList_R[idx].constraints = 0.0f;
	}

	for (int idx = 0; idx < bucketsize_T; idx++)
	{
		m_bucketList_T[idx].bucketIndex = idx;
		//	m_bucketTranslation[idx].clear();
		if (m_bucketTranslation[idx].empty())
			m_bucketList_T[idx].constraints = FLT_MAX;
		else
			m_bucketList_T[idx].constraints = 0.0f;
	}
	typeBucket r = typeBucket::Rotation, t = typeBucket::Translation;
	updateBucketOrder(r);
	updateBucketOrder(t);
}

void DNSS::sortIntoBucket()
{
	int nSizePoints = static_cast<int>(m_vertices_normalized.size());
	std::wcout << L"niszeP_points : " << nSizePoints << std::endl;
	int nSizeBucketR = m_bucketsizeR_azimuth * m_bucketsizeR_polar;
	int nSizeBucketT = m_bucketsizeT_azimuth * m_bucketsizeT_polar;
	// concurrent vector에 넣고 정렬해서 멤버 변수에 다시 넣는다.
	// 이유: concurrent_vector pop_back 사용불가
	std::vector<concurrency::concurrent_vector<std::pair<int, float>>> bucketRotation;
	std::vector<concurrency::concurrent_vector<int>> bucketTranslation;
	m_vecForBIdx.clear();
	m_vecForBIdx.resize(nSizePoints);

	bucketRotation.clear();
	bucketRotation.resize(nSizeBucketR);
	bucketTranslation.clear();
	bucketTranslation.resize(nSizeBucketT);

	m_bucketRotation.clear();
	m_bucketRotation.resize(nSizeBucketR);
	m_bucketTranslation.clear();
	m_bucketTranslation.resize(nSizeBucketT);


	//버켓 공간 확보 - 버켓에 완전히 균일하게 들어가는 것이 아니고 한쪽에 몰릴수도 있어, 버켓 사이즈로 나누지 않고 10으로 나눔(임의로 정한 수)
	Concurrency::parallel_for(0, nSizeBucketR, [&](int idx)
	{
		bucketRotation[idx].reserve(nSizePoints / 10);
	});
	Concurrency::parallel_for(0, nSizeBucketT, [&](int idx)
	{
		bucketTranslation[idx].reserve(nSizePoints / 10);
	});

	Concurrency::parallel_for(0, nSizePoints, [&](int idx)
	{
		float coord_R_azimuth = 0;
		float coord_R_polar = 0;
		computeSphericalCoordinate(glm::normalize(m_normals_rotational[idx]), coord_R_azimuth, coord_R_polar);
		int index_R_azimuth = static_cast<int>(std::floorf(fabs(coord_R_azimuth) / m_thetaForSort));
		int index_R_polar = static_cast<int>(std::floorf(coord_R_polar / m_thetaForSort));
		int index_R = index_R_azimuth * m_bucketsizeR_polar + index_R_polar;

		bucketRotation[index_R].push_back(std::make_pair(idx, m_rotationalReturns[idx]));

		float coord_T_azimuth = 0;
		float coord_T_polar = 0;
		computeSphericalCoordinate(glm::normalize(m_normals_original[idx]), coord_T_azimuth, coord_T_polar);
		int index_T_azimuth = static_cast<int>(std::floorf((coord_T_azimuth + m_pi_degree) / m_thetaForSort));
		int index_T_polar = static_cast<int>(std::floorf(coord_T_polar / m_thetaForSort));
		int index_T = index_T_azimuth * m_bucketsizeT_polar + index_T_polar;

		bucketTranslation[index_T].push_back(idx);

		m_vecForBIdx[idx] = std::make_pair(index_R, index_T);
	}
	);

	Concurrency::parallel_for(0, nSizeBucketR, [&](int idx)
	{
		bucketRotation[idx].shrink_to_fit();
		m_bucketRotation[idx].resize(bucketRotation[idx].size());
		std::move(bucketRotation[idx].begin(), bucketRotation[idx].end(), m_bucketRotation[idx].begin());

	});
	Concurrency::parallel_for(0, nSizeBucketT, [&](int idx)
	{
		bucketTranslation[idx].shrink_to_fit();
		m_bucketTranslation[idx].resize(bucketTranslation[idx].size());
		std::move(bucketTranslation[idx].begin(), bucketTranslation[idx].end(), m_bucketTranslation[idx].begin());
	});


	Concurrency::parallel_for(0, nSizeBucketR, [&](int idx)
	{
		//pop_back만 가능하므로 return값이 큰 것이 뒤로 가도록 정렬
		//ascendin order sort because only pop_back is possible
		std::sort(m_bucketRotation[idx].begin(), m_bucketRotation[idx].end(),
			[](const std::pair<int, float>& lhs, const std::pair<int, float>& rhs) {
			return lhs.second < rhs.second;
		});
	});




	Concurrency::parallel_for(0, nSizeBucketT, [&](int idx)
	{
		std::random_shuffle(m_bucketTranslation[idx].begin(), m_bucketTranslation[idx].end());
	});
}

void DNSS::computeRotationalNormals()
{
	int nSize = static_cast<int>(m_normals_original.size());
	m_normals_rotational.resize(m_normals_original.size());
	Concurrency::parallel_for(0, nSize, [&](int idx)
	{
		m_normals_rotational[idx] = glm::cross(m_vertices_normalized[idx], m_normals_original[idx]);
	});
}

void DNSS::computeCentroidandNormalize()
{
	Eigen::Matrix3Xf vertices = Eigen::Map<const Eigen::Matrix3Xf>(&m_vertices_original[0].x, 3, m_vertices_original.size());
	Eigen::Vector3f centroid_eigen = vertices.rowwise().mean();
	glm::fvec3 centroid = (glm::fvec3&)(*centroid_eigen.data());

	int nSize = static_cast<int>(m_vertices_original.size());
	m_vertices_normalized.clear();
	m_vertices_normalized.resize(nSize);
	Concurrency::parallel_for(0, nSize, [&](int idx)
	{
		m_vertices_normalized[idx] = m_vertices_original[idx] - centroid;
	});
	Eigen::Matrix3Xf vertices_moved = Eigen::Map<const Eigen::Matrix3Xf>(&m_vertices_normalized[0].x, 3, m_vertices_normalized.size());
	//normalize factor
	Eigen::Vector3f centroid_eigen_moved = vertices_moved.rowwise().maxCoeff();
	float Lmax = vertices_moved.colwise().norm().maxCoeff();
	float L_inverse = 1 / Lmax;
	Concurrency::parallel_for(0, nSize, [&](int idx)
	{
		m_vertices_normalized[idx] = m_vertices_normalized[idx] * L_inverse;
	});
}

int DNSS::pickPoint()
{
	int bid_top;
	typeBucket bType_top, bType_another;

	if (m_bucketList_R.front().constraints <= m_bucketList_T.front().constraints)
	{
		bid_top = m_bucketList_R.front().bucketIndex;
		bType_top = typeBucket::Rotation;
		bType_another = typeBucket::Translation;
	}
	else
	{
		bid_top = m_bucketList_T.front().bucketIndex;
		bType_top = typeBucket::Translation;
		bType_another = typeBucket::Rotation;
	}
	int pid = -1;

	switch (bType_top)
	{
	case typeBucket::Rotation:
	{
		if (m_bucketRotation[bid_top].empty())
		{
			std::wcout << L"bucket id : " << bid_top <<
				" and bucket size : " << m_bucketRotation[bid_top].size() << std::endl;
			std::wcout << L"constraints : " << m_bucketList_R[bid_top].constraints << std::endl;

		}

		pid = m_bucketRotation[bid_top].back().first;

		m_bucketList_R.front().constraints += m_bucketRotation[bid_top].back().second;
		m_bucketRotation[bid_top].pop_back();

		int bid_T = m_vecForBIdx[pid].second;

		if (!m_bucketTranslation[bid_T].empty())
		{
			m_bucketTranslation[bid_T].erase(std::remove_if(m_bucketTranslation[bid_T].begin(), m_bucketTranslation[bid_T].end(),
				[&pid](const int &elem) { return elem == pid; }),
				m_bucketTranslation[bid_T].end());
		}
		if (m_bucketTranslation[bid_T].empty())
		{
			auto it = std::find_if(m_bucketList_T.begin(), m_bucketList_T.end(),
				[bid_T](const structBucket & element) { return element.bucketIndex == bid_T; });
			it->constraints = FLT_MAX;
			updateBucketOrder(bType_another);
		}

		if (m_bucketRotation[bid_top].empty())
		{
			m_bucketList_R.front().constraints = FLT_MAX;
		}
		updateBucketOrder(bType_top);
		break;
	}
	case typeBucket::Translation:
	{
		if (m_bucketTranslation[bid_top].empty())
		{
			std::wcout << L"bucket id : " << bid_top <<
				" and bucket size : " << m_bucketTranslation[bid_top].size() << std::endl;
			std::wcout << L"constraints : " << m_bucketList_T[bid_top].constraints << std::endl;
		}
		pid = m_bucketTranslation[bid_top].back();


		m_bucketList_T.front().constraints += 1.0f;
		m_bucketTranslation[bid_top].pop_back();

		int bid_R = m_vecForBIdx[pid].first;

		if (!m_bucketRotation[bid_R].empty())
		{
			m_bucketRotation[bid_R].erase(std::remove_if(m_bucketRotation[bid_R].begin(), m_bucketRotation[bid_R].end(),
				[&pid](const std::pair<int, float> &elem) { return elem.first == pid; }),
				m_bucketRotation[bid_R].end());
		}
		if (m_bucketRotation[bid_R].empty())
		{
			auto it = std::find_if(m_bucketList_R.begin(), m_bucketList_R.end(),
				[bid_R](const structBucket & element) { return element.bucketIndex == bid_R; });
			it->constraints = FLT_MAX;
			updateBucketOrder(bType_another);
		}
		if (m_bucketTranslation[bid_top].empty())
		{
			m_bucketList_T.front().constraints = FLT_MAX;
		}
		updateBucketOrder(bType_top);
		break;
	}
	}
	return pid;
}
void DNSS::updateBucketOrder(typeBucket &bType)
{
	switch (bType)
	{
	case typeBucket::Rotation:
		std::sort(m_bucketList_R.begin(), m_bucketList_R.end(),
			[](const structBucket& lhs, const structBucket & rhs) {
			return lhs.constraints < rhs.constraints;
		});
		break;
	case typeBucket::Translation:
		std::sort(m_bucketList_T.begin(), m_bucketList_T.end(),
			[](const structBucket& lhs, const structBucket & rhs) {
			return lhs.constraints < rhs.constraints;
		});
		break;
	}

}

void DNSS::computeSphericalCoordinate(const glm::fvec3 &normal, float &coordinates_azimuth, float &coordinates_polar)
{
	float radian_azimuth = atan2(normal.y, normal.x);
	float radian_polar = acos(normal.z);
	coordinates_azimuth = radian_azimuth * m_pi_degree / m_pi_radian;
	coordinates_polar = radian_polar * m_pi_degree / m_pi_radian;

}
