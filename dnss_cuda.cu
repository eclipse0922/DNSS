#include <cmath>
#include <vector>

#include <cuda_runtime.h>

#include "NSS.h"

namespace
{
constexpr float kEpsilon = 1e-7f;

__device__ float clampFloat(float value, float min_value, float max_value)
{
	return fmaxf(min_value, fminf(value, max_value));
}

__global__ void computeRotationalFeaturesKernel(
	const float *normalized_vertices,
	const float *normals,
	float theta,
	float *rotational_normals,
	float *rotational_returns,
	int point_count)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= point_count)
	{
		return;
	}

	const int base = 3 * index;
	const float px = normalized_vertices[base + 0];
	const float py = normalized_vertices[base + 1];
	const float pz = normalized_vertices[base + 2];
	const float nx = normals[base + 0];
	const float ny = normals[base + 1];
	const float nz = normals[base + 2];

	// Rotational normal nr = p x n.
	rotational_normals[base + 0] = py * nz - pz * ny;
	rotational_normals[base + 1] = pz * nx - px * nz;
	rotational_normals[base + 2] = px * ny - py * nx;

	const float point_norm = sqrtf(px * px + py * py + pz * pz);
	if (!(isfinite(point_norm)) || point_norm <= kEpsilon)
	{
		rotational_returns[index] = 0.0f;
		return;
	}

	float normal_norm = sqrtf(nx * nx + ny * ny + nz * nz);
	if (!(isfinite(normal_norm)) || normal_norm <= kEpsilon)
	{
		normal_norm = 1.0f;
	}

	const float pdx = px / point_norm;
	const float pdy = py / point_norm;
	const float pdz = pz / point_norm;
	const float ndx = nx / normal_norm;
	const float ndy = ny / normal_norm;
	const float ndz = nz / normal_norm;

	const float dot_pn = clampFloat(pdx * ndx + pdy * ndy + pdz * ndz, -1.0f, 1.0f);
	const float beta = acosf(dot_pn);
	const float pp = 2.0f * point_norm * sinf(theta * 0.5f);
	const float sin_beta = sinf(beta);
	const float cos_beta = cosf(beta);

	const float pq_positive = pp * cosf(beta - theta * 0.5f);
	const float pq_negative = pp * cosf(-beta - theta * 0.5f);

	const float denominator_positive = fmaxf(kEpsilon, point_norm - pq_positive * cos_beta);
	const float denominator_negative = fmaxf(kEpsilon, point_norm - pq_negative * cos_beta);

	const float atan_positive = atanf((pq_positive * sin_beta) / denominator_positive);
	const float atan_negative = atanf((pq_negative * (-sin_beta)) / denominator_negative);

	const float gamma_positive = theta - atan_positive;
	const float gamma_negative = theta - atan_negative;

	const float rr_positive = point_norm * gamma_positive / theta;
	const float rr_negative = point_norm * gamma_negative / theta;

	const float rr = fmaxf(rr_positive, rr_negative);
	if (!isfinite(rr) || rr < 0.0f)
	{
		rotational_returns[index] = 0.0f;
	}
	else
	{
		rotational_returns[index] = rr;
	}
}

bool isCudaSuccess(cudaError_t status)
{
	return status == cudaSuccess;
}

} // namespace

bool DNSSComputeRotationalFeaturesCuda(
	const std::vector<glm::fvec3> &normalized_vertices,
	const std::vector<glm::fvec3> &normals,
	float theta_for_return_radians,
	std::vector<glm::fvec3> *rotational_normals,
	std::vector<float> *rotational_returns)
{
	if (rotational_normals == nullptr || rotational_returns == nullptr)
	{
		return false;
	}
	if (normalized_vertices.size() != normals.size())
	{
		return false;
	}

	const int point_count = static_cast<int>(normalized_vertices.size());
	rotational_normals->assign(normalized_vertices.size(), glm::fvec3(0.0f));
	rotational_returns->assign(normalized_vertices.size(), 0.0f);

	if (point_count == 0)
	{
		return true;
	}

	std::vector<float> vertices_flat(static_cast<std::size_t>(point_count) * 3u);
	std::vector<float> normals_flat(static_cast<std::size_t>(point_count) * 3u);
	for (int i = 0; i < point_count; ++i)
	{
		vertices_flat[3 * i + 0] = normalized_vertices[static_cast<std::size_t>(i)].x;
		vertices_flat[3 * i + 1] = normalized_vertices[static_cast<std::size_t>(i)].y;
		vertices_flat[3 * i + 2] = normalized_vertices[static_cast<std::size_t>(i)].z;
		normals_flat[3 * i + 0] = normals[static_cast<std::size_t>(i)].x;
		normals_flat[3 * i + 1] = normals[static_cast<std::size_t>(i)].y;
		normals_flat[3 * i + 2] = normals[static_cast<std::size_t>(i)].z;
	}

	float *d_vertices = nullptr;
	float *d_normals = nullptr;
	float *d_rot_normals = nullptr;
	float *d_rot_returns = nullptr;

	const std::size_t vector_bytes = vertices_flat.size() * sizeof(float);
	const std::size_t return_bytes = rotational_returns->size() * sizeof(float);

	if (!isCudaSuccess(cudaMalloc(&d_vertices, vector_bytes)) ||
		!isCudaSuccess(cudaMalloc(&d_normals, vector_bytes)) ||
		!isCudaSuccess(cudaMalloc(&d_rot_normals, vector_bytes)) ||
		!isCudaSuccess(cudaMalloc(&d_rot_returns, return_bytes)))
	{
		cudaFree(d_vertices);
		cudaFree(d_normals);
		cudaFree(d_rot_normals);
		cudaFree(d_rot_returns);
		return false;
	}

	if (!isCudaSuccess(cudaMemcpy(d_vertices, vertices_flat.data(), vector_bytes, cudaMemcpyHostToDevice)) ||
		!isCudaSuccess(cudaMemcpy(d_normals, normals_flat.data(), vector_bytes, cudaMemcpyHostToDevice)))
	{
		cudaFree(d_vertices);
		cudaFree(d_normals);
		cudaFree(d_rot_normals);
		cudaFree(d_rot_returns);
		return false;
	}

	const int threads = 256;
	const int blocks = (point_count + threads - 1) / threads;
	computeRotationalFeaturesKernel<<<blocks, threads>>>(
		d_vertices,
		d_normals,
		theta_for_return_radians,
		d_rot_normals,
		d_rot_returns,
		point_count);

	if (!isCudaSuccess(cudaGetLastError()) || !isCudaSuccess(cudaDeviceSynchronize()))
	{
		cudaFree(d_vertices);
		cudaFree(d_normals);
		cudaFree(d_rot_normals);
		cudaFree(d_rot_returns);
		return false;
	}

	std::vector<float> rot_normals_flat(vertices_flat.size());
	std::vector<float> rot_returns_flat(rotational_returns->size());

	if (!isCudaSuccess(cudaMemcpy(rot_normals_flat.data(), d_rot_normals, vector_bytes, cudaMemcpyDeviceToHost)) ||
		!isCudaSuccess(cudaMemcpy(rot_returns_flat.data(), d_rot_returns, return_bytes, cudaMemcpyDeviceToHost)))
	{
		cudaFree(d_vertices);
		cudaFree(d_normals);
		cudaFree(d_rot_normals);
		cudaFree(d_rot_returns);
		return false;
	}

	cudaFree(d_vertices);
	cudaFree(d_normals);
	cudaFree(d_rot_normals);
	cudaFree(d_rot_returns);

	for (int i = 0; i < point_count; ++i)
	{
		(*rotational_normals)[static_cast<std::size_t>(i)] = glm::fvec3(
			rot_normals_flat[3 * i + 0],
			rot_normals_flat[3 * i + 1],
			rot_normals_flat[3 * i + 2]);
		(*rotational_returns)[static_cast<std::size_t>(i)] = rot_returns_flat[static_cast<std::size_t>(i)];
	}

	return true;
}
