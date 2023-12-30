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
	/** \brief Function to perform normal space sampling
	 * \param[in] sample_rate The sampling rate, maximum 1.0. Repeat until the number is filled by multiplying the rate.
	 * \param[out] sampled_vertices Data of sampled points.
	 * \param[out] sampled_normals Data of sampled normals.
	 */
	void normalSpaceSampling(const float &sample_rate, std::vector<glm::fvec3> &sampled_vertices, std::vector<glm::fvec3> &sampled_normals);

	/** \brief Function to specify the original data for sampling
	 * \param[in] pointData Point data.
	 * \param[in] normalData Normal data.
	 */
	void setInputCloud(const std::vector<glm::fvec3> original_vertices, const std::vector<glm::fvec3> original_normals);

private:
	/** \brief Create indices based on the normal values and then store the index values in buckets.
	 * \param[in] const std::vector<glm::fvec3> &n A series of normal values.
	 * \param[out] std::vector< std::vector<int>> Buckets containing indices.
	 */
	void sort_into_buckets(const std::vector<glm::fvec3> &n, std::vector<std::vector<int>> &normbuckets);

private:
	std::vector<glm::fvec3> m_original_vertices;
	std::vector<glm::fvec3> m_original_normals;
};

/*
 * Dual Normal Space Sampling
 * Kwok, Tsz-Ho. "DNSS: Dual-Normal-Space Sampling for 3-D ICP Registration."
 * IEEE Transactions on Automation Science and Engineering (2018).
 * Sampling including rotation information in addition to the original NSS.
 * Adding Rotational Information to NSS
 * Author: seweon.jeon
 * Date: 2018.04.09
 **/

class DNSS
{
public:
	void setInputCloud(const std::vector<glm::fvec3> original_vertices, const std::vector<glm::fvec3> original_normals);
	void dualNormalSpaceSampling(const float &sample_rate, std::vector<glm::fvec3> &sampled_vertices, std::vector<glm::fvec3> &sampled_normals);

private:
	struct structBucket
	{
		int bucketIndex = 0;
		float constraints = 0;
	};

	enum class typeBucket : int
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
	 * \brief Calculates azimuth and polar values based on Normal (nx, ny, nz)
	 * \note The naming of azimuth and polar angles differ in the paper and ISO rule, so this follows the ISO rule
	 * ISO rule: theta - polar angle (0~ pi), phi - azimuth angle (0~2pi)
	 * Paper: theta - azimuth angle (0~2pi), phi - polar angle (0~pi)
	 * pi = 180; 2pi = 360;
	 * \param[in] glm::fvec3 Normal value.
	 * \param[out] float coordinates_azimuth.
	 * \param[out] float coordinates_polar.
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
	 * \note The naming of azimuth and polar angles differ in the paper and ISO rule, so this follows the ISO rule
	 * ISO rule: theta - polar angle (0~180), phi - azimuth angle (0~360)
	 * Paper: theta - azimuth angle (0~360), phi - polar angle (0~180)
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