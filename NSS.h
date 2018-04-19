#pragma once

#include <glm/glm.hpp>
#include <algorithm>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <concurrent_queue.h>
#include <concurrent_vector.h>
#include <concurrent_unordered_map.h>

/*
 * normal space sampling algorithm
*
 *
 * original source code: https://github.com/kyzyx/scanalyze/
 *Scanalyze (version 1.0) license information
 -------------------------------------------

 Scanalyze is copyright (c) 2002 the Board of Trustees of The Leland
 Stanford Junior University. All rights reserved.

 This software is covered by the Stanford Computer Graphics Laboratory's
 General Software License. This license is royalty-free, nonexclusive,
 and nontransferable.  The terms and conditions of this license are
 viewable at this URL:

 http://graphics.stanford.edu/software/license.html


 The scanalyze software also builds upon a number of 3rd party
 software components, some of which may be copyrighted:

 1. Togl
 2. Tcl/Tk
 3. STL
 4. Template Numerical Toolkit (TNT)
 5. Janne Heikkila's Camera Calibration Toolbox
 6. Graphics Gems
 7. Miscellaneous SGI code

 Further information about each of these software components and
 their terms of usage is included below:

 ----------------------------------------------------------------------
 *
 **/
class NSS
{
public:

	/** \brief normal space sampling 수행 함수
	* \param[in] sample_rate 샘플링레이트  최대 1.0  비율 곱해서 개수 채워질때까지 반복수행
	* \param[out] sampled_vertices 샘플링한 포인트 데이터
	* \param[out] sampled_normals 샘플링한 노멀 데이터
	*/
	void normalSpaceSampling(const float &sample_rate, std::vector<glm::fvec3> &sampled_vertices, std::vector<glm::fvec3> &sampled_normals);

	/** \brief sampling하기위한 원본 데이터 지정위한 함수
	* \param[in] pointData 포인트 데이터
	* \param[in] normalData 노멀 데이터
	*/
	void setInputCloud(const std::vector<glm::fvec3> original_vertices, const std::vector<glm::fvec3> original_normals)
	{
		m_original_vertices = original_vertices;
		m_original_normals = original_normals;
	}
private:
	/** \brief normal 값을 기준으로 인덱스 생성한 뒤 인덱스 값을 보고 버켓에 저장
	* \param[in] const std::vector<glm::fvec3> &n 일련의 노말(normal) 값
	* \param[out] std::vector< std::vector<int>> 인덱스가 들어간 버켓
	*/
	void sort_into_buckets(const std::vector<glm::fvec3> &n, std::vector< std::vector<int>> &normbuckets);

private:
	std::vector<glm::fvec3> m_original_vertices;
	std::vector<glm::fvec3> m_original_normals;
};

/*
 * Dual Normal Space Sampling
 * Kwok, Tsz-Ho. "DNSS: Dual-Normal-Space Sampling for 3-D ICP Registration."
 * IEEE Transactions on Automation Science and Engineering (2018).
 * 기존 NSS에 rotation 정보도 포함시켜서 샘플링
 * adding Rotational Information to NSS
 * 직접 구현
 * 작성자:  seweon.jeon
 * 2018.04.09
 **/

class DNSS
{
public:
	void setInputCloud(const std::vector<glm::fvec3> original_vertices, const std::vector<glm::fvec3> original_normals)
	{
		m_vertices_original = original_vertices;
		m_normals_original = original_normals;
	}
	void dualNormalSpaceSampling(const float &sample_rate, std::vector<glm::fvec3> &sampled_vertices, std::vector<glm::fvec3> &sampled_normals);
private:
	struct structBucket
	{
		int bucketIndex = 0;
		float constraints = 0;
	};

	enum class typeBucket :int
	{
		Rotation = 0,
		Translation,
	};

	void computeRotationalReturn();
	void initBucketB();
	void sortIntoBucket();
	void computeRotationalNormals();
	void computeCentroidandNormalize();

	int pickPoint();

	void updateBucketOrder(typeBucket &bType);
	/**
	 * \brief Normal(nx,ny,nz) 받아서 azimuth,polar 값계산
	 *   논문과 iso rule 각각에서 azimuth, polar angle  지칭하는 것이 달라 iso rule에 맞춤
	 *     i followed ISO rule for spherical coordinate system naming
	 *  iso rule    theta -  polar angle   (0~ pi) phi - azimuth angle( 0~2pi)
	 *  논문         theta -  azimuth angle (0~2pi) phi - polar angle  ( 0~pi)
	 *  pi== 180; 2pi ==360;
	 * \param[in] glm::fvec3 normal 값
	 * \param[out] float coordinates_theta
	 * \param[out] float coordinates_phi
	 */
	void computeSphericalCoordinate(const glm::fvec3 &normal, float &coordinates_azimuth, float &coordinates_polar);

private:
	std::vector<structBucket> m_bucketList_R;
	std::vector<structBucket> m_bucketList_T;
	std::vector<std::pair<int, int>> m_vecForBIdx;
	std::vector<std::vector<std::pair<int, float>>> m_bucketRotation;
	std::vector<std::vector<int>> m_bucketTranslation;
	std::vector<glm::fvec3> m_vertices_original;
	std::vector<glm::fvec3> m_normals_original;
	std::vector<glm::fvec3> m_vertices_normalized;
	std::vector<glm::fvec3> m_normals_rotational;
	std::vector<float> m_rotationalReturns;

	/*
	*
	 *   \note 논문과 iso rule 각각에서 azimuth, polar angle 지칭하는 것이 달라 iso rule에 맞춤
	 *      i followed ISO rule for spherical coordinate system naming   
	 *		iso rule    theta -  polar angle   (0~180) phi - azimuth angle( 0~360)
	 *		paper       theta -  azimuth angle (0~360) phi - polar angle( 0~180)
	 */
	static const int m_bucketsizeT_azimuth = 12;
	static const int m_bucketsizeT_polar = 6;
	static const int m_bucketsizeR_azimuth = 6;
	static const int m_bucketsizeR_polar = 6;

	const float m_pi_degree = 180.0f;
	const float m_pi_radian = 3.141592f;
	const float m_pi2_degree = 360.0f;
	const float m_thetaForSort = m_pi_degree / 6;
	const float m_thetaForReturn = m_pi_degree / 4;
};

