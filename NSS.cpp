#include "NSS.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>

namespace
{
constexpr float kPi = 3.14159265359f;
constexpr float kTwoPi = 2.0f * kPi;
constexpr float kEpsilon = 1e-7f;

float clampFloat(float value, float min_value, float max_value)
{
	return std::max(min_value, std::min(value, max_value));
}

float dotProduct(const glm::fvec3 &lhs, const glm::fvec3 &rhs)
{
	return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

glm::fvec3 crossProduct(const glm::fvec3 &lhs, const glm::fvec3 &rhs)
{
	return glm::fvec3(
		lhs.y * rhs.z - lhs.z * rhs.y,
		lhs.z * rhs.x - lhs.x * rhs.z,
		lhs.x * rhs.y - lhs.y * rhs.x);
}

float vectorLength(const glm::fvec3 &value)
{
	return std::sqrt(dotProduct(value, value));
}

glm::fvec3 normalizeSafe(const glm::fvec3 &value)
{
	const float norm = vectorLength(value);
	if (!std::isfinite(norm) || norm <= kEpsilon)
	{
		return glm::fvec3(0.0f, 0.0f, 1.0f);
	}
	return value / norm;
}

float computeRotationalReturnValue(const glm::fvec3 &normalized_point, const glm::fvec3 &normal, float theta)
{
	const float point_norm = vectorLength(normalized_point);
	if (!std::isfinite(point_norm) || point_norm <= kEpsilon)
	{
		return 0.0f;
	}

	const glm::fvec3 point_dir = normalized_point / point_norm;
	const glm::fvec3 normal_dir = normalizeSafe(normal);
	const float dot_pn = clampFloat(dotProduct(point_dir, normal_dir), -1.0f, 1.0f);
	const float beta = std::acos(dot_pn);

	const float pp = 2.0f * point_norm * std::sin(theta * 0.5f);
	const float sin_beta = std::sin(beta);
	const float cos_beta = std::cos(beta);

	const float pq_positive = pp * std::cos(beta - theta * 0.5f);
	const float pq_negative = pp * std::cos(-beta - theta * 0.5f);

	const float denominator_positive = std::max(kEpsilon, point_norm - pq_positive * cos_beta);
	const float denominator_negative = std::max(kEpsilon, point_norm - pq_negative * cos_beta);

	const float atan_positive = std::atan((pq_positive * sin_beta) / denominator_positive);
	const float atan_negative = std::atan((pq_negative * (-sin_beta)) / denominator_negative);

	const float gamma_positive = theta - atan_positive;
	const float gamma_negative = theta - atan_negative;

	const float rr_positive = point_norm * gamma_positive / theta;
	const float rr_negative = point_norm * gamma_negative / theta;
	const float rotational_return = std::max(rr_positive, rr_negative);

	if (!std::isfinite(rotational_return))
	{
		return 0.0f;
	}
	return std::max(0.0f, rotational_return);
}

int sphericalBucketIndex(
	const glm::fvec3 &raw_normal,
	int azimuth_bins,
	int z_bins,
	bool fold_antipodal)
{
	if (azimuth_bins <= 0 || z_bins <= 0)
	{
		return -1;
	}

	glm::fvec3 normal = normalizeSafe(raw_normal);

	if (fold_antipodal)
	{
		const bool flip =
			normal.z < 0.0f ||
			(normal.z == 0.0f && (normal.y < 0.0f || (normal.y == 0.0f && normal.x < 0.0f)));
		if (flip)
		{
			normal = -normal;
		}
	}

	float azimuth = std::atan2(normal.y, normal.x);
	if (azimuth < 0.0f)
	{
		azimuth += kTwoPi;
	}

	const float z_min = fold_antipodal ? 0.0f : -1.0f;
	const float z_normalized = clampFloat((normal.z - z_min) / (1.0f - z_min), 0.0f, 1.0f);

	int azimuth_index = static_cast<int>(std::floor(azimuth / kTwoPi * static_cast<float>(azimuth_bins)));
	int z_index = static_cast<int>(std::floor(z_normalized * static_cast<float>(z_bins)));

	if (azimuth_index >= azimuth_bins)
	{
		azimuth_index = azimuth_bins - 1;
	}
	if (z_index >= z_bins)
	{
		z_index = z_bins - 1;
	}

	return azimuth_index * z_bins + z_index;
}

void computeCenteredAndScaledVertices(
	const std::vector<glm::fvec3> &vertices,
	std::vector<glm::fvec3> &normalized_vertices)
{
	normalized_vertices.clear();
	normalized_vertices.resize(vertices.size(), glm::fvec3(0.0f));
	if (vertices.empty())
	{
		return;
	}

	glm::fvec3 centroid(0.0f, 0.0f, 0.0f);
	for (const glm::fvec3 &vertex : vertices)
	{
		centroid += vertex;
	}
	centroid /= static_cast<float>(vertices.size());

	float max_radius = 0.0f;
	for (std::size_t i = 0; i < vertices.size(); ++i)
	{
		normalized_vertices[i] = vertices[i] - centroid;
		max_radius = std::max(max_radius, vectorLength(normalized_vertices[i]));
	}

	if (max_radius <= kEpsilon)
	{
		return;
	}

	const float inv_radius = 1.0f / max_radius;
	for (glm::fvec3 &vertex : normalized_vertices)
	{
		vertex *= inv_radius;
	}
}

} // namespace

#if defined(DNSS_HAS_CUDA)
bool DNSSComputeRotationalFeaturesCuda(
	const std::vector<glm::fvec3> &normalized_vertices,
	const std::vector<glm::fvec3> &normals,
	float theta_for_return_radians,
	std::vector<glm::fvec3> *rotational_normals,
	std::vector<float> *rotational_returns);
#endif

NSS::NSS()
	: NSS(Options{})
{
}

NSS::NSS(const Options &options)
	: m_options(options), m_runtime_seed(options.random_seed)
{
}

void NSS::setOptions(const Options &options)
{
	m_options = options;
	m_runtime_seed = options.random_seed;
}

const NSS::Options &NSS::getOptions() const
{
	return m_options;
}

void NSS::setInputCloud(const std::vector<glm::fvec3> &original_vertices, const std::vector<glm::fvec3> &original_normals)
{
	if (original_vertices.size() != original_normals.size())
	{
		throw std::invalid_argument("NSS::setInputCloud requires matching vertex and normal counts.");
	}
	m_original_vertices = original_vertices;
	m_original_normals = original_normals;
}

void NSS::normalSpaceSampling(const float &sample_rate, std::vector<glm::fvec3> &sampled_vertices, std::vector<glm::fvec3> &sampled_normals)
{
	sampled_vertices.clear();
	sampled_normals.clear();

	const int target_count = targetSampleCount(sample_rate);
	if (target_count <= 0)
	{
		return;
	}

	const int bucket_count = m_options.azimuth_bins * m_options.z_bins;
	if (bucket_count <= 0)
	{
		throw std::invalid_argument("NSS bucket configuration must be positive.");
	}

	std::vector<std::vector<int>> buckets(static_cast<std::size_t>(bucket_count));
	for (std::size_t index = 0; index < m_original_normals.size(); ++index)
	{
		int bucket_index = normalToBucketIndex(m_original_normals[index]);
		if (bucket_index < 0 || bucket_index >= bucket_count)
		{
			bucket_index = 0;
		}
		buckets[static_cast<std::size_t>(bucket_index)].push_back(static_cast<int>(index));
	}

	std::mt19937 rng(m_options.deterministic ? m_options.random_seed : nextSeed());
	for (std::vector<int> &bucket : buckets)
	{
		std::shuffle(bucket.begin(), bucket.end(), rng);
	}

	sampled_vertices.reserve(static_cast<std::size_t>(target_count));
	sampled_normals.reserve(static_cast<std::size_t>(target_count));

	while (sampled_vertices.size() < static_cast<std::size_t>(target_count))
	{
		bool selected_any = false;
		for (std::vector<int> &bucket : buckets)
		{
			if (sampled_vertices.size() >= static_cast<std::size_t>(target_count))
			{
				break;
			}
			if (bucket.empty())
			{
				continue;
			}

			selected_any = true;
			const int point_index = bucket.back();
			bucket.pop_back();

			sampled_vertices.push_back(m_original_vertices[static_cast<std::size_t>(point_index)]);
			sampled_normals.push_back(m_original_normals[static_cast<std::size_t>(point_index)]);
		}

		if (!selected_any)
		{
			break;
		}
	}
}

int NSS::normalToBucketIndex(const glm::fvec3 &normal) const
{
	return sphericalBucketIndex(normal, m_options.azimuth_bins, m_options.z_bins, false);
}

int NSS::targetSampleCount(float sample_rate) const
{
	if (!std::isfinite(sample_rate) || sample_rate <= 0.0f || m_original_vertices.empty())
	{
		return 0;
	}

	const float clamped_rate = std::min(1.0f, sample_rate);
	const int count = static_cast<int>(std::ceil(clamped_rate * static_cast<float>(m_original_vertices.size())));
	return std::clamp(count, 0, static_cast<int>(m_original_vertices.size()));
}

std::uint32_t NSS::nextSeed()
{
	m_runtime_seed = m_runtime_seed * 1664525u + 1013904223u;
	return m_runtime_seed;
}

DNSS::DNSS()
	: DNSS(Options{})
{
}

DNSS::DNSS(const Options &options)
	: m_options(options), m_runtime_seed(options.random_seed)
{
}

void DNSS::setOptions(const Options &options)
{
	m_options = options;
	m_runtime_seed = options.random_seed;
}

const DNSS::Options &DNSS::getOptions() const
{
	return m_options;
}

void DNSS::setUseCuda(bool enable)
{
	m_options.enable_cuda = enable;
}

bool DNSS::getUseCuda() const
{
	return m_options.enable_cuda;
}

void DNSS::setInputCloud(const std::vector<glm::fvec3> &original_vertices, const std::vector<glm::fvec3> &original_normals)
{
	if (original_vertices.size() != original_normals.size())
	{
		throw std::invalid_argument("DNSS::setInputCloud requires matching vertex and normal counts.");
	}
	m_vertices_original = original_vertices;
	m_normals_original = original_normals;
}

void DNSS::dualNormalSpaceSampling(const float &sample_rate, std::vector<glm::fvec3> &sampled_vertices, std::vector<glm::fvec3> &sampled_normals)
{
	sampled_vertices.clear();
	sampled_normals.clear();

	const int target_count = targetSampleCount(sample_rate);
	if (target_count <= 0)
	{
		return;
	}

	const int t_bucket_count = m_options.t_azimuth_bins * m_options.t_z_bins;
	const int r_bucket_count = m_options.r_azimuth_bins * m_options.r_z_bins;
	if (t_bucket_count <= 0 || r_bucket_count <= 0)
	{
		throw std::invalid_argument("DNSS bucket configuration must be positive.");
	}

	std::vector<glm::fvec3> normalized_vertices;
	computeCenteredAndScaledVertices(m_vertices_original, normalized_vertices);

	std::vector<glm::fvec3> rotational_normals(m_normals_original.size(), glm::fvec3(0.0f));
	std::vector<float> rotational_returns(m_normals_original.size(), 0.0f);

	bool cuda_used = false;
#if defined(DNSS_HAS_CUDA)
	if (m_options.enable_cuda)
	{
		cuda_used = DNSSComputeRotationalFeaturesCuda(
			normalized_vertices,
			m_normals_original,
			m_options.theta_for_return_radians,
			&rotational_normals,
			&rotational_returns);
	}
#endif

	if (!cuda_used)
	{
		for (std::size_t i = 0; i < normalized_vertices.size(); ++i)
		{
			rotational_normals[i] = crossProduct(normalized_vertices[i], m_normals_original[i]);
			rotational_returns[i] = computeRotationalReturnValue(
				normalized_vertices[i],
				m_normals_original[i],
				m_options.theta_for_return_radians);
		}
	}

	std::vector<std::vector<int>> t_buckets(static_cast<std::size_t>(t_bucket_count));

	struct RotationEntry
	{
		float rotational_return;
		int point_index;
	};
	struct RotationEntryComparator
	{
		bool operator()(const RotationEntry &lhs, const RotationEntry &rhs) const
		{
			return lhs.rotational_return < rhs.rotational_return;
		}
	};

	using RotationQueue = std::priority_queue<RotationEntry, std::vector<RotationEntry>, RotationEntryComparator>;
	std::vector<RotationQueue> r_buckets(static_cast<std::size_t>(r_bucket_count));

	std::vector<int> point_to_t_bucket(m_vertices_original.size(), 0);
	std::vector<int> point_to_r_bucket(m_vertices_original.size(), 0);

	for (std::size_t point_index = 0; point_index < m_vertices_original.size(); ++point_index)
	{
		int t_bucket = translationalBucketIndex(m_normals_original[point_index]);
		int r_bucket = rotationalBucketIndex(rotational_normals[point_index]);

		if (t_bucket < 0 || t_bucket >= t_bucket_count)
		{
			t_bucket = 0;
		}
		if (r_bucket < 0 || r_bucket >= r_bucket_count)
		{
			r_bucket = 0;
		}

		t_buckets[static_cast<std::size_t>(t_bucket)].push_back(static_cast<int>(point_index));
		r_buckets[static_cast<std::size_t>(r_bucket)].push(
			RotationEntry{rotational_returns[point_index], static_cast<int>(point_index)});

		point_to_t_bucket[point_index] = t_bucket;
		point_to_r_bucket[point_index] = r_bucket;
	}

	std::mt19937 rng(m_options.deterministic ? m_options.random_seed : nextSeed());
	for (std::vector<int> &bucket : t_buckets)
	{
		std::shuffle(bucket.begin(), bucket.end(), rng);
	}

	std::vector<float> t_constraints(static_cast<std::size_t>(t_bucket_count), 0.0f);
	std::vector<float> r_constraints(static_cast<std::size_t>(r_bucket_count), 0.0f);
	std::vector<unsigned char> is_active(m_vertices_original.size(), 1u);

	sampled_vertices.reserve(static_cast<std::size_t>(target_count));
	sampled_normals.reserve(static_cast<std::size_t>(target_count));

	auto popTranslationalPoint = [&](int bucket_index) -> int
	{
		auto &bucket = t_buckets[static_cast<std::size_t>(bucket_index)];
		while (!bucket.empty() && is_active[static_cast<std::size_t>(bucket.back())] == 0u)
		{
			bucket.pop_back();
		}
		if (bucket.empty())
		{
			return -1;
		}
		const int point_index = bucket.back();
		bucket.pop_back();
		return point_index;
	};

	auto popRotationalPoint = [&](int bucket_index, float *rotational_weight) -> int
	{
		auto &bucket = r_buckets[static_cast<std::size_t>(bucket_index)];
		while (!bucket.empty() && is_active[static_cast<std::size_t>(bucket.top().point_index)] == 0u)
		{
			bucket.pop();
		}
		if (bucket.empty())
		{
			*rotational_weight = 0.0f;
			return -1;
		}
		const RotationEntry entry = bucket.top();
		bucket.pop();
		*rotational_weight = entry.rotational_return;
		return entry.point_index;
	};

	auto selectPoint = [&](int point_index)
	{
		if (point_index < 0)
		{
			return;
		}
		if (is_active[static_cast<std::size_t>(point_index)] == 0u)
		{
			return;
		}

		is_active[static_cast<std::size_t>(point_index)] = 0u;
		sampled_vertices.push_back(m_vertices_original[static_cast<std::size_t>(point_index)]);
		sampled_normals.push_back(m_normals_original[static_cast<std::size_t>(point_index)]);

		const int t_bucket = point_to_t_bucket[static_cast<std::size_t>(point_index)];
		const int r_bucket = point_to_r_bucket[static_cast<std::size_t>(point_index)];
		t_constraints[static_cast<std::size_t>(t_bucket)] += 1.0f;
		r_constraints[static_cast<std::size_t>(r_bucket)] += rotational_returns[static_cast<std::size_t>(point_index)];
	};

	if (m_options.initialize_rotation_buckets)
	{
		for (int r_bucket = 0; r_bucket < r_bucket_count; ++r_bucket)
		{
			if (sampled_vertices.size() >= static_cast<std::size_t>(target_count))
			{
				break;
			}
			float weight = 0.0f;
			const int point_index = popRotationalPoint(r_bucket, &weight);
			(void)weight;
			selectPoint(point_index);
		}
	}

	while (sampled_vertices.size() < static_cast<std::size_t>(target_count))
	{
		enum class BucketType
		{
			None,
			Translational,
			Rotational
		};

		BucketType best_type = BucketType::None;
		int best_bucket_index = -1;
		float best_constraint = std::numeric_limits<float>::max();

		for (int t_bucket = 0; t_bucket < t_bucket_count; ++t_bucket)
		{
			auto &bucket = t_buckets[static_cast<std::size_t>(t_bucket)];
			while (!bucket.empty() && is_active[static_cast<std::size_t>(bucket.back())] == 0u)
			{
				bucket.pop_back();
			}
			if (bucket.empty())
			{
				continue;
			}

			const float constraint = t_constraints[static_cast<std::size_t>(t_bucket)];
			if (constraint < best_constraint)
			{
				best_constraint = constraint;
				best_bucket_index = t_bucket;
				best_type = BucketType::Translational;
			}
		}

		for (int r_bucket = 0; r_bucket < r_bucket_count; ++r_bucket)
		{
			auto &bucket = r_buckets[static_cast<std::size_t>(r_bucket)];
			while (!bucket.empty() && is_active[static_cast<std::size_t>(bucket.top().point_index)] == 0u)
			{
				bucket.pop();
			}
			if (bucket.empty())
			{
				continue;
			}

			const float constraint = r_constraints[static_cast<std::size_t>(r_bucket)];
			if (constraint < best_constraint)
			{
				best_constraint = constraint;
				best_bucket_index = r_bucket;
				best_type = BucketType::Rotational;
			}
		}

		if (best_type == BucketType::None)
		{
			break;
		}

		if (best_type == BucketType::Translational)
		{
			selectPoint(popTranslationalPoint(best_bucket_index));
		}
		else
		{
			float rotational_weight = 0.0f;
			selectPoint(popRotationalPoint(best_bucket_index, &rotational_weight));
		}
	}

	if (sampled_vertices.size() < static_cast<std::size_t>(target_count))
	{
		for (std::size_t point_index = 0; point_index < is_active.size(); ++point_index)
		{
			if (sampled_vertices.size() >= static_cast<std::size_t>(target_count))
			{
				break;
			}
			if (is_active[point_index] == 0u)
			{
				continue;
			}
			selectPoint(static_cast<int>(point_index));
		}
	}
}

int DNSS::translationalBucketIndex(const glm::fvec3 &normal) const
{
	return sphericalBucketIndex(normal, m_options.t_azimuth_bins, m_options.t_z_bins, false);
}

int DNSS::rotationalBucketIndex(const glm::fvec3 &rotational_normal) const
{
	return sphericalBucketIndex(rotational_normal, m_options.r_azimuth_bins, m_options.r_z_bins, true);
}

int DNSS::targetSampleCount(float sample_rate) const
{
	if (!std::isfinite(sample_rate) || sample_rate <= 0.0f || m_vertices_original.empty())
	{
		return 0;
	}

	const float clamped_rate = std::min(1.0f, sample_rate);
	const int count = static_cast<int>(std::ceil(clamped_rate * static_cast<float>(m_vertices_original.size())));
	return std::clamp(count, 0, static_cast<int>(m_vertices_original.size()));
}

std::uint32_t DNSS::nextSeed()
{
	m_runtime_seed = m_runtime_seed * 1664525u + 1013904223u;
	return m_runtime_seed;
}
