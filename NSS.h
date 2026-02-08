#pragma once

#include <cstdint>
#include <vector>

#if __has_include(<glm/vec3.hpp>)
#include <glm/vec3.hpp>
#else
namespace glm
{
struct fvec3
{
	float x;
	float y;
	float z;

	constexpr fvec3()
		: x(0.0f), y(0.0f), z(0.0f)
	{
	}

	constexpr explicit fvec3(float value)
		: x(value), y(value), z(value)
	{
	}

	constexpr fvec3(float x_value, float y_value, float z_value)
		: x(x_value), y(y_value), z(z_value)
	{
	}

	float &operator[](int index)
	{
		return (&x)[index];
	}

	const float &operator[](int index) const
	{
		return (&x)[index];
	}

	fvec3 &operator+=(const fvec3 &rhs)
	{
		x += rhs.x;
		y += rhs.y;
		z += rhs.z;
		return *this;
	}

	fvec3 &operator-=(const fvec3 &rhs)
	{
		x -= rhs.x;
		y -= rhs.y;
		z -= rhs.z;
		return *this;
	}

	fvec3 &operator*=(float scalar)
	{
		x *= scalar;
		y *= scalar;
		z *= scalar;
		return *this;
	}

	fvec3 &operator/=(float scalar)
	{
		x /= scalar;
		y /= scalar;
		z /= scalar;
		return *this;
	}
};

inline fvec3 operator+(fvec3 lhs, const fvec3 &rhs)
{
	lhs += rhs;
	return lhs;
}

inline fvec3 operator-(fvec3 lhs, const fvec3 &rhs)
{
	lhs -= rhs;
	return lhs;
}

inline fvec3 operator-(const fvec3 &value)
{
	return fvec3(-value.x, -value.y, -value.z);
}

inline fvec3 operator*(fvec3 lhs, float scalar)
{
	lhs *= scalar;
	return lhs;
}

inline fvec3 operator*(float scalar, fvec3 rhs)
{
	rhs *= scalar;
	return rhs;
}

inline fvec3 operator/(fvec3 lhs, float scalar)
{
	lhs /= scalar;
	return lhs;
}
} // namespace glm
#endif

class NSS
{
public:
	struct Options
	{
		int azimuth_bins = 12;
		int z_bins = 6;
		std::uint32_t random_seed = 0x12345678u;
		bool deterministic = true;
	};

	NSS();
	explicit NSS(const Options &options);

	void setOptions(const Options &options);
	const Options &getOptions() const;

	/** \brief Function to perform normal space sampling.
	 * \param[in] sample_rate The sampling rate in [0, 1].
	 * \param[out] sampled_vertices Data of sampled points.
	 * \param[out] sampled_normals Data of sampled normals.
	 */
	void normalSpaceSampling(const float &sample_rate, std::vector<glm::fvec3> &sampled_vertices, std::vector<glm::fvec3> &sampled_normals);

	/** \brief Function to specify the original data for sampling.
	 * \param[in] original_vertices Point data.
	 * \param[in] original_normals Normal data.
	 */
	void setInputCloud(const std::vector<glm::fvec3> &original_vertices, const std::vector<glm::fvec3> &original_normals);

private:
	int normalToBucketIndex(const glm::fvec3 &normal) const;
	int targetSampleCount(float sample_rate) const;
	std::uint32_t nextSeed();

private:
	Options m_options;
	std::uint32_t m_runtime_seed = m_options.random_seed;
	std::vector<glm::fvec3> m_original_vertices;
	std::vector<glm::fvec3> m_original_normals;
};

class DNSS
{
public:
	struct Options
	{
		int t_azimuth_bins = 12;
		int t_z_bins = 6;
		int r_azimuth_bins = 6;
		int r_z_bins = 6;
		float theta_for_return_radians = 0.78539816339f; // pi / 4
		bool initialize_rotation_buckets = true;
		bool enable_cuda = false;
		std::uint32_t random_seed = 0x9e3779b9u;
		bool deterministic = true;
	};

	DNSS();
	explicit DNSS(const Options &options);

	void setOptions(const Options &options);
	const Options &getOptions() const;

	void setUseCuda(bool enable);
	bool getUseCuda() const;

	void setInputCloud(const std::vector<glm::fvec3> &original_vertices, const std::vector<glm::fvec3> &original_normals);
	void dualNormalSpaceSampling(const float &sample_rate, std::vector<glm::fvec3> &sampled_vertices, std::vector<glm::fvec3> &sampled_normals);

private:
	int translationalBucketIndex(const glm::fvec3 &normal) const;
	int rotationalBucketIndex(const glm::fvec3 &rotational_normal) const;
	int targetSampleCount(float sample_rate) const;
	std::uint32_t nextSeed();

private:
	Options m_options;
	std::uint32_t m_runtime_seed = m_options.random_seed;
	std::vector<glm::fvec3> m_vertices_original;
	std::vector<glm::fvec3> m_normals_original;
};
