/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

#include "gc/shared/gc_globals.hpp"
#include "gc/shared/gcLogPrecious.hpp"
#include "gc/z/zAdaptiveHeap.inline.hpp"
#include "gc/z/zDriver.hpp"
#include "gc/z/zHeap.inline.hpp"
#include "gc/z/zLock.inline.hpp"
#include "gc/z/zStat.hpp"
#include "logging/log.hpp"
#include "runtime/atomicAccess.hpp"
#include "runtime/globals_extension.hpp"
#include "runtime/os.hpp"
#include "utilities/debug.hpp"
#include "utilities/globalDefinitions.hpp"

#include <cmath>
#include <limits>

bool ZAdaptiveHeap::_explicit_max_capacity;
bool ZAdaptiveHeap::_can_adapt;
bool ZAdaptiveHeap::_initialized;
TruncatedSeq ZAdaptiveHeap::_gc_intensities;

Atomic<double> ZAdaptiveHeap::_young_to_old_gc_time(0.0);
double ZAdaptiveHeap::_accumulated_young_gc_time = 0.0;
ZAdaptiveHeap::ZGenerationOverhead ZAdaptiveHeap::_young_data;
ZAdaptiveHeap::ZGenerationOverhead ZAdaptiveHeap::_old_data;
Atomic<uint> ZAdaptiveHeap::_initial_young_worker_cap;
ZAdaptiveHeap::ZSqrtSizerState ZAdaptiveHeap::_sqrt_young_state;
ZAdaptiveHeap::ZSqrtSizerState ZAdaptiveHeap::_sqrt_old_state;

static ZLock* _stat_lock;

void ZAdaptiveHeap::initialize(bool explicit_max_capacity, bool can_adapt) {
  precond(!_initialized);
  double process_time_now = os::elapsed_process_cpu_time();
  double time_now = os::elapsedTime();
  _young_data._last_machine_system_time = process_time_now;
  _old_data._last_machine_system_time = process_time_now;
  _young_data._last_container_system_time = process_time_now;
  _old_data._last_container_system_time = process_time_now;
  _young_data._last_process_time = process_time_now;
  _old_data._last_process_time = process_time_now;
  _young_data._last_time = time_now;
  _old_data._last_time = time_now;
  _explicit_max_capacity = explicit_max_capacity;
  _can_adapt = can_adapt;
  _stat_lock = new ZLock();
  _initialized = true;
}

double ZAdaptiveHeap::young_to_old_gc_time() {
  precond(_initialized);
  return _young_to_old_gc_time.load_relaxed();
}

uint ZAdaptiveHeap::initial_young_worker_cap() {
  precond(_initialized);
  uint capacity = _initial_young_worker_cap.load_relaxed();
  if (capacity == 0) {
    // Not yet set; use one - there are barely any objects early on anyway
    return 1;
  }

  return capacity;
}

static double* generate_factorials(uint64_t max) {
  double* result = NEW_C_HEAP_ARRAY(double, max + 1, mtGC);
  result[0] = 1.0;
  for (uint64_t i = 1; i <= max; ++i) {
    result[i] = result[i - 1] * i;
  }
  return result;
}

// Probability to have to wait in a queue given c servers and rho queue utilization
static double erlang_c(int c, double rho, int max_processors) {
  static double* factorials = nullptr;
  if (factorials == nullptr) {
    factorials = generate_factorials(max_processors);
  }

  double unutilized_reciprocal = 1.0 / (1.0 - rho);
  double offered_load = rho * c;
  double nominator = unutilized_reciprocal * pow(offered_load, double(c)) / factorials[c];

  double sum = 0.0;
  for (int k = 0; k < c; ++k) {
    sum += pow(offered_load, double(k)) / factorials[k];
  }

  double denominator = nominator + sum;

  return nominator / denominator;
}

static double cpu_latency_factor(int c, double rho, double unluckyness, int max_processors) {
  double prob_join_queue = erlang_c(c, rho, max_processors);

  // P99 is approximately 100x more likely to join the queue
  double p99_prob_join_queue = MIN2(prob_join_queue * unluckyness, 1.0);
  double p99_latency_factor = p99_prob_join_queue / (1.0 - rho);

  return 1.0 + p99_latency_factor;
}

ZMemoryPressureMetrics ZAdaptiveHeap::memory_pressure_metrics() {
  precond(_initialized);
  const double unscaled_gc_intensity = AtomicAccess::load(&ZGCIntensity);

  const physical_memory_size_type machine_max_memory = os::Machine::physical_memory();

  physical_memory_size_type machine_used_memory;
  physical_memory_size_type machine_compressed_memory;
  if (!os::Machine::used_memory(machine_used_memory)) {
    // Approximation for faulty OS
    assert(false, "Why would this ever happen?");
    machine_used_memory = os::rss();
  }
  if (os::compressed_memory(machine_compressed_memory)) {
    machine_compressed_memory = MIN2(machine_compressed_memory, machine_used_memory);
  } else {
    machine_compressed_memory = 0;
  }

  // The concerning threshold is after which memory utilization we start trying
  // harder to keep the memory down. There are multiple reasons for letting the GC
  // run hotter:
  // 1) We want to maintain some headroom on the machine so that we can deal with
  //    spikes without getting allocation stalls.
  // 2) It's good to let the OS keep some file system cache memory around
  // 3) On systems that compress used memory, using compressed memory is not a
  //    free lunch as it leads to page faults that compress and decompress memory.
  //    This is extra painful for a tracing GC to traverse.
  const double machine_compression_rate = double(machine_compressed_memory) / double(machine_max_memory);

  // When the machine memory usage is fluctuating and isn't stable, it's good to
  // increase the headroom to the memory limits, otherwise there is a higher risk
  // of an allocation stall.
  const double machine_memory_instability = ZStatSystemMemoryUsage::machine_memory_stability();

  const double memory_uncertainty = MAX2(machine_memory_instability, machine_compression_rate);
  const double machine_concerning_threshold = MIN2(ZMemoryConcerningThreshold + memory_uncertainty, 1.0);

  const double machine_concerning_vs_high_diff = ZMemoryConcerningThreshold - ZMemoryHighThreshold;
  const double machine_high_threshold = machine_concerning_threshold - machine_concerning_vs_high_diff;

  const double machine_critical_threshold = ZMemoryCriticalThreshold;

  const double far_avoid = 1.0 - ZMemoryConcerningThreshold;
  const double medium_avoid = 1.0 - ZMemoryHighThreshold;
  const double near_avoid = 1.0 - ZMemoryCriticalThreshold;

  physical_memory_size_type container_max_memory;
  physical_memory_size_type container_used_memory;
  double container_concerning_threshold;
  double container_high_threshold;
  double container_critical_threshold;

  const bool can_read_container_memory = os::is_containerized() && os::Container::used_memory(container_used_memory);
  bool has_container_limit = false;
  if (can_read_container_memory) {
    physical_memory_size_type container_critical_memory;

    // Allocation stalls at critical levels
    if (os::Container::memory_limit(container_max_memory)) {
      has_container_limit = true;
    } else {
      container_max_memory = machine_max_memory;
    }

    // Exponential increase in pressure up to critical
    physical_memory_size_type container_high_memory;
    if (os::Container::memory_throttle_limit(container_high_memory)) {
      container_max_memory = MIN2(container_max_memory, container_high_memory);
      container_critical_memory = container_max_memory;
      container_high_memory = MIN2(container_high_memory, container_critical_memory) * (medium_avoid / near_avoid);
      has_container_limit = true;
    } else {
      container_critical_memory = near_avoid * container_max_memory;
      container_high_memory = container_critical_memory * (medium_avoid / near_avoid);
    }

    // Linear increase in pressure up to intensity squared
    physical_memory_size_type container_min_memory;
    if (os::Container::memory_soft_limit(container_min_memory)) {
      container_min_memory = MIN2(container_min_memory, physical_memory_size_type(container_high_memory * (far_avoid / medium_avoid)));
      has_container_limit = true;
    } else {
      container_min_memory = physical_memory_size_type(container_high_memory * (far_avoid / medium_avoid));
    }

    container_critical_threshold = 1.0 - double(container_critical_memory) / double(container_max_memory);
    container_high_threshold = 1.0 - double(container_high_memory) / double(container_max_memory);
    container_concerning_threshold = 1.0 - double(container_min_memory) / double(container_max_memory);

    // Adjust the threshold by how unstable the memory usage in the system is
    const double container_memory_instability = ZStatSystemMemoryUsage::container_memory_stability();
    const double instability_adjustment = MIN2(container_memory_instability, 1.0 - container_concerning_threshold);
    container_concerning_threshold += instability_adjustment;
    container_high_threshold += instability_adjustment;
  }

  bool is_containerized = can_read_container_memory && has_container_limit;

  if (!is_containerized) {
    container_max_memory = machine_max_memory;
    container_used_memory = machine_used_memory;
    container_concerning_threshold = machine_concerning_threshold;
    container_high_threshold = machine_high_threshold;
    container_critical_threshold = machine_critical_threshold;
  }

  // Record sampled memory usage so we can measure memory stability
  const double machine_usage = double(machine_used_memory) / double(machine_max_memory);
  ZStatSystemMemoryUsage::record_machine_usage(machine_usage);

  if (is_containerized) {
    const double container_usage = double(container_used_memory) / double(container_max_memory);
    ZStatSystemMemoryUsage::record_container_usage(container_usage);
  }

  return {
    unscaled_gc_intensity,
    is_containerized,
    {
      machine_used_memory,
      machine_max_memory,
      machine_concerning_threshold,
      machine_high_threshold,
      machine_critical_threshold
    },
    {
      container_used_memory,
      container_max_memory,
      container_concerning_threshold,
      container_high_threshold,
      container_critical_threshold
    }
  };
}

static double system_memory_pressure(const ZSystemMemoryPressureMetrics& metrics, double unscaled_gc_intensity) {
  const physical_memory_size_type available_memory = metrics._max_memory - metrics._used_memory;

  // The remaining memory reserve of the machine
  const double availability = double(available_memory) / double(metrics._max_memory);

  // A number indicating how much the memory pressure should grow as the
  // memory unavailability grows
  const double pressure_rate = MAX2(unscaled_gc_intensity, 2.0);

  const double concerning = metrics._concerning_threshold;
  const double high = metrics._high_threshold;
  const double critical = metrics._critical_threshold;

  if (availability < high) {
    // When memory pressure is "high", we exponentially scale up memory pressure,
    // from the already "high" pressure induced by "concerning" memory pressure.
    const double progression = 1.0 - (availability - critical) / (high - critical);
    const double exponent = 1.0 + progression;
    return pow(pressure_rate, exponent);
  }

  if (availability < concerning) {
    // When memory pressure is "concerning", we linearly scale up memory pressure to the
    // "high" pressure (i.e. unscaled GC intensity).
    const double progression = 1.0 - (availability - high) / (concerning - high);
    const double pressure_factor = (pressure_rate - 1.0) * progression;

    return 1.0 + pressure_factor;
  }

  return 1.0;
}

double ZAdaptiveHeap::compute_memory_pressure(const ZMemoryPressureMetrics& metrics) {
  const double machine_mem_pressure = system_memory_pressure(metrics._machine, metrics._unscaled_gc_intensity);

  if (!metrics._is_containerized) {
    return machine_mem_pressure;
  }

  const double container_mem_pressure = system_memory_pressure(metrics._container, metrics._unscaled_gc_intensity);
  return MAX2(machine_mem_pressure, container_mem_pressure);
}

// Calculate progression from 0 to 1 going down from a less critical availability to a more critical availability
static double calculate_progression(double availability, double from, double to) {
  return 1.0 - (availability - to) / (from - to);
}

static bool is_system_memory_pressure_concerning(const ZSystemMemoryPressureMetrics& metrics) {
  const physical_memory_size_type available_memory = metrics._max_memory - metrics._used_memory;
  const double availability = double(available_memory) / double(metrics._max_memory);

  return availability < metrics._concerning_threshold;
}

bool ZAdaptiveHeap::is_memory_pressure_concerning(const ZMemoryPressureMetrics& metrics) {
  if (is_system_memory_pressure_concerning(metrics._machine)) {
    return true;
  }

  if (metrics._is_containerized) {
    return is_system_memory_pressure_concerning(metrics._container);
  }

  return false;
}

static bool is_system_memory_pressure_high(const ZSystemMemoryPressureMetrics& metrics) {
  const physical_memory_size_type available_memory = metrics._max_memory - metrics._used_memory;
  const double availability = double(available_memory) / double(metrics._max_memory);

  return availability < metrics._high_threshold;
}

bool ZAdaptiveHeap::is_memory_pressure_high(const ZMemoryPressureMetrics& metrics) {
  if (is_system_memory_pressure_high(metrics._machine)) {
    return true;
  }

  if (metrics._is_containerized) {
    return is_system_memory_pressure_high(metrics._container);
  }

  return false;
}

static bool is_system_memory_pressure_critical(const ZSystemMemoryPressureMetrics& metrics) {
  const physical_memory_size_type available_memory = metrics._max_memory - metrics._used_memory;
  const double availability = double(available_memory) / double(metrics._max_memory);

  return availability < metrics._critical_threshold;
}

bool ZAdaptiveHeap::is_memory_pressure_critical(const ZMemoryPressureMetrics& metrics) {
  if (is_system_memory_pressure_critical(metrics._machine)) {
    return true;
  }

  if (metrics._is_containerized) {
    return is_system_memory_pressure_critical(metrics._container);
  }

  return false;
}

ZCpuPressureMetrics ZAdaptiveHeap::cpu_pressure_metrics(ZGenerationId generation) {
  const bool is_young = generation == ZGenerationId::young;
  ZGenerationOverhead& generation_data = is_young ? _young_data : _old_data;

  // Time metrics
  const double machine_system_time_last = generation_data._last_machine_system_time;
  const double container_system_time_last = generation_data._last_container_system_time;
  // Note that the system time might have poor accuracy early on; it typically
  // has 100 ms granularity. So take it with a large grain of salt early on...
  const double machine_system_time_now = os::Machine::elapsed_system_cpu_time();
  const double machine_system_time = machine_system_time_now - machine_system_time_last;

  const int machine_ncpus = os::Machine::active_processor_count();

  double container_system_time_now;
  double container_ncpus;

  const bool can_read_container_usage = os::is_containerized() && os::Container::elapsed_system_cpu_time(container_system_time_now);
  bool has_container_limit = false;
  if (can_read_container_usage) {
    const double container_system_time = container_system_time_now - container_system_time_last;
    generation_data._container_system_times.add(container_system_time);

    if (os::Container::processor_count(container_ncpus) && container_ncpus < double(machine_ncpus)) {
      has_container_limit = true;
    } else {
      container_ncpus = machine_ncpus;
    }
  }

  const bool is_containerized = can_read_container_usage && has_container_limit;

  const double time_now = os::elapsedTime();
  const double time_last = generation_data._last_time;
  const double time_since_last = time_now - time_last;
  generation_data._last_machine_system_time = machine_system_time_now;
  generation_data._last_container_system_time = container_system_time_now;
  generation_data._last_time = time_now;

  if (!is_containerized) {
    container_system_time_now = machine_system_time_now;
    container_ncpus = machine_ncpus;
  }

  generation_data._machine_system_times.add(machine_system_time);

  ZStatCycleStats cycle_stats = ZGeneration::generation(generation)->stat_cycle()->stats();
  const double gc_time = cycle_stats._last_total_vtime + (is_young ? 0.0 : _accumulated_young_gc_time);

  generation_data._gc_times.add(gc_time);
  generation_data._gc_times_since_last.add(time_since_last);

  const double avg_gc_time = generation_data._gc_times.avg();
  const double avg_time_since_last = generation_data._gc_times_since_last.avg();
  const double avg_machine_system_time = generation_data._machine_system_times.avg();
  const double avg_container_system_time = generation_data._container_system_times.avg();

  // Process times
  const double process_time_last = generation_data._last_process_time;
  const double process_time_now = os::elapsed_process_cpu_time();
  const double process_time = process_time_now - process_time_last;
  generation_data._last_process_time = process_time_now;
  generation_data._process_times.add(process_time);
  const double avg_process_time = generation_data._process_times.avg();
  const double avg_generation_gc_cpu_overhead = avg_gc_time / avg_process_time;
  const double generation_gc_cpu_overhead = gc_time / process_time;

  const double avg_machine_process_cpu_load = clamp((avg_process_time / avg_time_since_last) / machine_ncpus, 0.0, 1.0);
  const double avg_machine_system_cpu_load = clamp((avg_machine_system_time / avg_time_since_last) / machine_ncpus, 0.0, 1.0);
  const double avg_container_process_cpu_load = clamp((avg_process_time / avg_time_since_last) / container_ncpus, 0.0, 1.0);
  const double avg_container_system_cpu_load = clamp((avg_container_system_time / avg_time_since_last) / container_ncpus, 0.0, 1.0);

  // Account for the overhead of old generation collections when evaluating
  // the heap efficiency for young generation collections.
  const double avg_total_gc_cpu_overhead = MIN2(avg_generation_gc_cpu_overhead / (is_young ? _young_to_old_gc_time.load_relaxed() : 1.0), 1.0);

  return {
    is_containerized,
    generation_gc_cpu_overhead,
    avg_generation_gc_cpu_overhead,
    avg_total_gc_cpu_overhead,
    avg_time_since_last,
    avg_process_time,
    gc_time,
    {
      avg_machine_process_cpu_load,
      avg_machine_system_cpu_load
    },
    {
      avg_container_process_cpu_load,
      avg_container_system_cpu_load
    }
  };
}

static double mem_urgency_scaled_cpu_pressure(const ZSystemMemoryPressureMetrics& mem_metrics, double cpu_pressure) {
  if (cpu_pressure >= 1.0) {
    return cpu_pressure;
  }

  // As memory pressure gets high, heap must be pressed down
  if (is_system_memory_pressure_high(mem_metrics)) {
    return 1.0;
  }

  // As memory pressure gets higher, heap must be pressed down
  if (!is_system_memory_pressure_concerning(mem_metrics)) {
    return cpu_pressure;
  }

  // Calculate concerning progression towards high
  const double availability = 1.0 - double(mem_metrics._used_memory) / double(mem_metrics._max_memory);
  const double progression = calculate_progression(availability, mem_metrics._concerning_threshold, mem_metrics._high_threshold);

  // Scale back to 1 as we approach high mem pressure
  return cpu_pressure + (1.0 - cpu_pressure) * progression;
}

static double compute_cpu_vs_memory_pressure(const ZSystemMemoryPressureMetrics& mem_metrics, const ZSystemCpuPressureMetrics& cpu_metrics, physical_memory_size_type process_used_memory) {
  const double process_memory_usage_ratio = clamp(double(process_used_memory) / double(mem_metrics._used_memory), 0.0, 1.0);

  const double process_cpu_usage_ratio = cpu_metrics._avg_process_load / cpu_metrics._avg_system_load;

  // The GC intensity is scaled by the relationship of how many of the system's
  // used bytes belong to this process compared to how many of the used system
  // CPU ticks belong to this process. For a single application deployment this
  // has effectively no effect, while for a multi process deployment, processes
  // that are unproportionately memory bloated compared to other processes will
  // rebalance themselves better to provide more memory for other processes.
  const double process_cpu_pressure = process_memory_usage_ratio - process_cpu_usage_ratio;

  // The GC intensity is scaled by what portion of system CPU resources are being
  // used. As CPU utilization of the machine gets higher, there will be more
  // fighting between mutator threads for CPU time, affecting latencies.
  // This concern is ignored here - this heuristic is purely reasoning about
  // the CPU vs memory resource availabilities. A separate cpu_vs_latency
  // calculation estimates the latency impact of too high CPU, and adjusts.
  const double system_cpu_usage = cpu_metrics._avg_system_load;
  const double system_memory_usage = double(mem_metrics._used_memory) / double(mem_metrics._max_memory);
  const double system_cpu_pressure = system_memory_usage - system_cpu_usage;

  // Balance the forces of resource share imbalance across processes with the
  // forces of system level resource usage imbalance.
  const double cpu_vs_memory_pressure = 1.0 + clamp(process_cpu_pressure + system_cpu_pressure, -0.2, 2.0);

  // Make sure that as memory availability drops, memory pressure starts to
  // dominate the overall GC intensity; without memory the JVM dies.
  return mem_urgency_scaled_cpu_pressure(mem_metrics, cpu_vs_memory_pressure);
}

static double compute_cpu_vs_memory_pressure(const ZMemoryPressureMetrics& mem_metrics, const ZCpuPressureMetrics& cpu_metrics, physical_memory_size_type process_used_memory) {
  const double machine_cpu_pressure = compute_cpu_vs_memory_pressure(mem_metrics._machine, cpu_metrics._machine, process_used_memory);

  if (!mem_metrics._is_containerized && !cpu_metrics._is_containerized) {
    return machine_cpu_pressure;
  }

  const double container_cpu_pressure = compute_cpu_vs_memory_pressure(mem_metrics._container, cpu_metrics._container, process_used_memory);
  return ::sqrt(container_cpu_pressure * machine_cpu_pressure);
}

static double compute_cpu_vs_latency_pressure(const ZSystemCpuPressureMetrics& machine_cpu_metrics) {
  // Approximate latency risks on the machine level using an M/M/c queue system.
  // This allows us to calculate by what factor tail latency will be affected by
  // the CPU pressure. Rather than looking at the current CPU pressure, we look
  // ahead a bit so we deal with latency problems proactively - before they arise,
  // instead of post mortem.
  // This type of M/M/c calculations make sense on a system level, but for the
  // container level, things work differently, and CPU vs memory is considered
  // purely as a resource balancing exercise, which still helps latency as well.
  double rho = MIN2(machine_cpu_metrics._avg_system_load + 0.05, 0.99);
  const double p99_response_time_scaling = cpu_latency_factor(os::Machine::active_processor_count(), rho, 100.0, os::processor_count());
  return 1.0 / MIN2(p99_response_time_scaling, 5.0);
}

static double compute_cpu_vs_latency_pressure(const ZMemoryPressureMetrics& mem_metrics, const ZCpuPressureMetrics& cpu_metrics) {
  const double machine_cpu_vs_latency_pressure = compute_cpu_vs_latency_pressure(cpu_metrics._machine);

  // Make sure that as machine memory availability drops, memory pressure starts to
  // dominate the overall GC intensity; without memory the JVM dies.
  const double machine_scaled_pressure = mem_urgency_scaled_cpu_pressure(mem_metrics._machine, machine_cpu_vs_latency_pressure);

  if (!mem_metrics._is_containerized && !cpu_metrics._is_containerized) {
    return machine_scaled_pressure;
  }

  // Make sure that as container availability drops, memory pressure starts to
  // dominate the overall GC intensity; without memory the JVM dies.
  // Note that using the machine CPU vs latency pressure here is intentional;
  // the probability of getting placed in a run queue is on machine level,
  // not on the container level.
  const double container_scaled_pressure = mem_urgency_scaled_cpu_pressure(mem_metrics._container, machine_cpu_vs_latency_pressure);

  return sqrt(machine_scaled_pressure * container_scaled_pressure);
}

ZResourcePressure ZAdaptiveHeap::compute_pressures(const ZMemoryPressureMetrics& mem_metrics, const ZCpuPressureMetrics& cpu_metrics, size_t projected_process_used_memory) {
  precond(_initialized);
  const double mem_pressure = compute_memory_pressure(mem_metrics);
  const double cpu_vs_memory_pressure = compute_cpu_vs_memory_pressure(mem_metrics, cpu_metrics, projected_process_used_memory);
  const double cpu_vs_latency_pressure = compute_cpu_vs_latency_pressure(mem_metrics, cpu_metrics);

  // Use sqrt to compute geometric mean
  const double cpu_pressure = ::sqrt(cpu_vs_memory_pressure * cpu_vs_latency_pressure);

  // The combined forces of memory vs CPU. The one force... TO RULE THEM ALL!!
  const double resource_pressure = mem_pressure * cpu_pressure;

  const double scaled_gc_intensity = mem_metrics._unscaled_gc_intensity * resource_pressure;
  double gc_intensity;

  {
    ZLocker<ZLock> locker(_stat_lock);
    _gc_intensities.add(scaled_gc_intensity);
    gc_intensity = MAX2(_gc_intensities.avg(), scaled_gc_intensity);
  }

  return {
    gc_intensity,
    cpu_pressure,
    mem_pressure,
    cpu_vs_memory_pressure,
    cpu_vs_latency_pressure
  };
}

static double compute_target_gc_interval(const ZSystemMemoryPressureMetrics& mem_metrics,
                                         const double gc_intensity) {
  // As memory pressure gets high, heap must be pressed down
  if (is_system_memory_pressure_high(mem_metrics)) {
    return 0.0;
  }

  // High GC frequencies lead to extra overheads such as barrier storms and
  // frequent cache invalidation of the heap.
  // Therefore, we add a factor that ensures there is at least some social
  // distancing between GCs, even when the GC overhead is small. The size of
  // the factor scales with the level of load induced on the machine, as well
  // as the scaled and unscaled GC intensities.
  const double gc_interval = 5.0 / gc_intensity;

  // As memory pressure gets higher, heap must be pressed down
  if (is_system_memory_pressure_concerning(mem_metrics)) {
    // Calculate concerning progression towards high
    const double availability = 1.0 - double(mem_metrics._used_memory) / double(mem_metrics._max_memory);
    const double progression = calculate_progression(availability, mem_metrics._concerning_threshold, mem_metrics._high_threshold);

    return (1.0 - progression) * gc_interval;
  }

  return gc_interval;
}

static double compute_target_gc_interval(const ZMemoryPressureMetrics& mem_metrics,
                                         const double gc_intensity) {
  const double machine_target_gc_interval = compute_target_gc_interval(mem_metrics._machine,
                                                                       gc_intensity);

  if (!mem_metrics._is_containerized) {
    return machine_target_gc_interval;
  }

  const double container_target_gc_interval = compute_target_gc_interval(mem_metrics._container,
                                                                         gc_intensity);
  return MIN2(machine_target_gc_interval, container_target_gc_interval);
}

// Logistic function, produces values in the range 0 - 1 in an S shape
static double sigmoid_function(double value) {
  return 1.0 / (1.0 + pow(M_E, -value));
}

// This function smoothens out measured error signals to make the incremental heap
// sizing converge better. During an initial warmup period, a more aggressive function
// is used, which doesn't try to reduce the error signals. This reduces the number of
// early GCs before the system has had any chance to converge to a stable heap size.
static double smoothing_function(double value, double warmness) {
  const double sigmoid = sigmoid_function(value);
  const double aggressive = MIN2(MAX2(sigmoid, 0.5 + value), 2.0);

  return sigmoid * warmness + aggressive * (1.0 - warmness);
}

// Square-root heap sizing policy based on:
//   "Optimal Heap Limits for Reducing Browser Memory Use" (OOPSLA 2022)
//   https://dl.acm.org/doi/10.1145/3563323
//
// The paper derives that the heap limit minimising the product of average
// memory use and GC overhead is:
//
//   M = L + sqrt(L * g / (c * s))
//
// where L is live bytes after GC, g is the allocation rate [bytes/s], c is
// the target GC CPU overhead fraction, and s is GC throughput [bytes/CPU-s].
//
// Substituting s = L / T_gc (GC throughput in terms of CPU time) the L terms
// cancel, giving the equivalent headroom:
//
//   E_sqrt = sqrt(g * T_gc * MiB / c)
//
// The MiB factor (1 048 576) is required for dimensional consistency: the
// paper works in MB units so its formula produces a result in MB; converting
// all inputs to raw bytes requires multiplying the sqrt argument by 1 MiB so
// the result is in bytes.  Without it the headroom is ~1024x too small.
//
// Exponentially weighted moving averages are applied to all three inputs
// following the smoothing parameters recommended by the paper:
//   alpha_alloc = 0.95  (~13-second half-life, tracks allocation bursts slowly)
//   alpha_gc    = 0.50  (faster response to GC-time changes)
//
// The same ZGCIntensity knob controls the target GC overhead c:
//   c = ZGCIntensity / 40.0   (consistent with the pressure-based algorithm)
//
// This function is called once per generation after every pressure-triggered
// GC cycle and returns a new total-heap heuristic capacity.
//
// ── EWMA state update (shared with Policy 2) ─────────────────────────────
void ZAdaptiveHeap::update_sqrt_state(ZSqrtSizerState& state,
                                      size_t live, double gc_cpu_time, double alloc_rate) {
  const double alpha_alloc = 0.95;
  const double alpha_gc    = 0.50;

  if (state._avg_live_bytes == 0.0) {
    // First call: seed directly so the estimate isn't dragged toward zero.
    state._avg_live_bytes  = double(live);
    state._avg_gc_cpu_time = gc_cpu_time;
    state._avg_alloc_rate  = alloc_rate;
  } else {
    state._avg_live_bytes  = alpha_gc    * state._avg_live_bytes  + (1.0 - alpha_gc)    * double(live);
    state._avg_gc_cpu_time = alpha_gc    * state._avg_gc_cpu_time + (1.0 - alpha_gc)    * gc_cpu_time;
    state._avg_alloc_rate  = alpha_alloc * state._avg_alloc_rate  + (1.0 - alpha_alloc) * alloc_rate;
  }
}

size_t ZAdaptiveHeap::compute_heap_size_sqrt(ZHeapResizeMetrics* heap_metrics,
                                              ZGenerationId generation) {
  ZStatCycleStats cycle_stats = ZGeneration::generation(generation)->stat_cycle()->stats();

  // L: live bytes for this generation at mark end.
  const size_t live = ZGeneration::generation(generation)->stat_heap()->live_at_mark_end();

  // T_gc: total CPU time consumed by this generation's last GC cycle.
  const double gc_cpu_time = cycle_stats._last_total_vtime;

  // g: mutator allocation rate [bytes / wall-clock second].
  const double alloc_rate = heap_metrics->_alloc_rate;

  ZSqrtSizerState& state = (generation == ZGenerationId::young)
      ? _sqrt_young_state
      : _sqrt_old_state;

  update_sqrt_state(state, live, gc_cpu_time, alloc_rate);

  // c: target GC CPU overhead — same knob as the pressure-based algorithm.
  const double target_gc_overhead = AtomicAccess::load(&ZGCIntensity) / 40.0;

  // E_sqrt = sqrt(g * T_gc * MiB / c), with a 2 MB floor from the paper.
  const size_t e_min = 2 * M;
  size_t extra = e_min;
  if (state._avg_gc_cpu_time > 0.0 && target_gc_overhead > 0.0) {
    const double E = sqrt(state._avg_alloc_rate * state._avg_gc_cpu_time * double(M) / target_gc_overhead);
    if (size_t(E) > extra) {
      extra = size_t(E);
    }
  }

  const size_t new_capacity = align_up(size_t(state._avg_live_bytes) + extra, ZGranuleSize);
  const size_t upper_bound  = MIN2(heap_metrics->_soft_max_capacity, heap_metrics->_current_max_capacity);
  const size_t lower_bound  = heap_metrics->_static_min_capacity;
  const size_t result       = clamp(new_capacity, lower_bound, upper_bound);

  if (can_adapt()) {
    log_info(gc, heap)("Sqrt: Live: %zuM, AllocRate: %.1fMB/s, GC CPU: %.3fs, Extra: %zuM -> Capacity: %zuM",
                       size_t(state._avg_live_bytes) / M,
                       state._avg_alloc_rate / M,
                       state._avg_gc_cpu_time,
                       extra / M,
                       result / M);
  }

  return result;
}

// Policy 2: combined sqrt + stall-prevention floor + GC-frequency floor (Idea G+).
//
// Preserves the Kirisame multi-tenant property: all three headroom terms depend
// only on the JVM's own (g, T_gc, L, c) — not on system RAM — so the shared
// constant c still acts as a global price signal across JVMs.
//
// Three terms compete to set the headroom:
//
//   E_sqrt  = sqrt(g * T_gc_cpu * MiB / c)
//             — Kirisame CPU-efficiency target (units-fixed).
//             Optimises GC CPU overhead across a shared machine.
//
//   E_stall = g * T_gc_wall
//             — stall-prevention floor: headroom needed to absorb one GC
//               cycle's worth of allocation without a mutator stall.
//               T_gc_wall = T_gc_cpu / active_workers + serial_time.
//               Dominates for workloads with g/s > 1 (large live set, slow GC).
//
//   E_floor = g * min(T_gc_cpu, T_base) / c,   T_base = c * T_min (50 ms)
//             — GC-frequency floor: analogous to E_freq but with T_gc_cpu
//               capped at T_base = c * T_min_interval.
//
//               Without the cap, E_freq = g * T_gc_cpu / c overshoots badly
//               for workloads with long GC CPU time (e.g. h2: T_gc_cpu ≈ 800ms
//               → E_freq ≈ 15 GB for a 32 GB machine, forcing max heap).
//
//               With T_min = 50 ms, T_base = 0.125 * 50ms = 6.25 ms:
//                 h2:       T_gc_cpu=800ms > T_base=6.25ms → E_floor=123 MB
//                           E_stall=2260 MB dominates → old behaviour preserved
//                 luindex:  T_gc_cpu=7ms   > T_base=6.25ms → E_floor=133 MB
//                           floor dominates → Linux fix (was 21 MB without it)
//
//               E_floor is multi-tenant safe: proportional to g_i with shared c.
//
// headroom = max(E_sqrt, E_stall, E_floor, 2 MB)
// M        = L + headroom
//
// Typical dominance:
//   g/s > 1, few workers (macOS): E_stall dominates.
//   g/s ≤ 1, many workers (Linux x86): E_floor dominates.
//   Low-activity workloads:  2 MB floor.
size_t ZAdaptiveHeap::compute_heap_size_combined(ZHeapResizeMetrics* heap_metrics,
                                                  ZGenerationId generation) {
  ZStatCycleStats cycle_stats = ZGeneration::generation(generation)->stat_cycle()->stats();

  const size_t live      = ZGeneration::generation(generation)->stat_heap()->live_at_mark_end();
  const double gc_cpu_time  = cycle_stats._last_total_vtime;
  const double alloc_rate   = heap_metrics->_alloc_rate;

  ZSqrtSizerState& state = (generation == ZGenerationId::young)
      ? _sqrt_young_state
      : _sqrt_old_state;

  update_sqrt_state(state, live, gc_cpu_time, alloc_rate);

  const double target_gc_overhead = AtomicAccess::load(&ZGCIntensity) / 40.0;

  // E_sqrt: units-fixed Kirisame headroom.
  const size_t e_min = 2 * M;
  size_t e_sqrt = e_min;
  if (state._avg_gc_cpu_time > 0.0 && target_gc_overhead > 0.0) {
    const double E = sqrt(state._avg_alloc_rate * state._avg_gc_cpu_time * double(M) / target_gc_overhead);
    if (size_t(E) > e_sqrt) {
      e_sqrt = size_t(E);
    }
  }

  // E_stall: allocation absorbed during one GC wall-clock cycle.
  // T_gc_wall = T_gc_cpu / active_workers + serial_time.
  const double active_workers = MAX2(cycle_stats._last_active_workers, 1.0);
  const double gc_wall_time   = state._avg_gc_cpu_time / active_workers
                              + cycle_stats._avg_serial_time;
  const size_t e_stall = size_t(state._avg_alloc_rate * gc_wall_time);

  // E_floor: GC-frequency floor with capped T_gc_cpu.
  // On many-core machines T_gc_wall << T_gc_cpu, so E_stall becomes negligible
  // even though hundreds of major GC cycles per second can occur (e.g. luindex
  // on Linux x86 with 8 workers: T_gc_wall=7ms/8=0.875ms → E_stall≈4 MB).
  //
  // Naïve fix E_freq = g * T_gc_cpu / c overshoots badly for workloads with
  // long GC CPU time: h2 has T_gc_cpu≈800ms → E_freq≈15 GB on a 32 GB machine.
  //
  // E_floor = g * min(T_gc_cpu, T_base) / c,  T_base = c * T_min_interval
  //
  // T_base = c * T_min caps the effective GC time so that:
  //   - Slow-GC workloads (T_gc_cpu > T_base): capped at T_base, preventing
  //     over-allocation. h2: T_gc_cpu=800ms > T_base=62.5ms → T_eff=62.5ms.
  //   - Fast-GC workloads (T_gc_cpu < T_base): uncapped, full E_floor applies.
  //     luindex Linux (8 workers): T_gc_cpu≈57ms < T_base=62.5ms → T_eff=57ms
  //     → E_floor = g*57ms/c ≈ 256 MB (vs E_stall≈4 MB → floor dominates). ✓
  //     luindex macOS (2 workers): T_gc_cpu≈7ms < T_base=62.5ms → T_eff=7ms
  //     → E_floor = g*7ms/c ≈ 28 MB (minimal change from old 6.25ms cap). ✓
  //
  // T_min = 500ms chosen so that T_base = 62.5ms sits just above the ~57ms
  // GC CPU time of fast-GC/many-core workloads on 8-core Linux, letting them
  // use their real T_gc_cpu without the cap.
  //
  // Multi-tenant safe: E_floor = g_i * min(T_cpu_i, T_base) / c, proportional
  // to each JVM's own workload with the shared constant c.
  const double T_min_interval = 0.500;  // 500 ms: T_base = c * 500ms = 62.5ms
  const double T_base  = target_gc_overhead * T_min_interval;
  const double T_eff   = MIN2(state._avg_gc_cpu_time, T_base);
  const size_t e_floor = (target_gc_overhead > 0.0)
      ? size_t(state._avg_alloc_rate * T_eff / target_gc_overhead)
      : size_t(0);

  const size_t extra        = MAX3(e_sqrt, e_stall, e_floor);
  const size_t new_capacity = align_up(size_t(state._avg_live_bytes) + extra, ZGranuleSize);
  const size_t upper_bound  = MIN2(heap_metrics->_soft_max_capacity, heap_metrics->_current_max_capacity);
  const size_t lower_bound  = heap_metrics->_static_min_capacity;
  const size_t result       = clamp(new_capacity, lower_bound, upper_bound);

  if (can_adapt()) {
    const char* dominant = (e_floor >= e_stall && e_floor >= e_sqrt) ? "floor"
                         : (e_stall >= e_sqrt)                      ? "stall"
                                                                    : "sqrt";
    log_info(gc, heap)("Combined [%s]: Live: %zuM, AllocRate: %.1fMB/s, "
                       "GC CPU: %.3fs, Wall: %.3fs, Extra: %zuM -> Capacity: %zuM",
                       dominant,
                       size_t(state._avg_live_bytes) / M,
                       state._avg_alloc_rate / M,
                       state._avg_gc_cpu_time,
                       gc_wall_time,
                       extra / M,
                       result / M);
  }

  return result;
}

size_t ZAdaptiveHeap::compute_heap_size(ZHeapResizeMetrics* heap_metrics, ZGenerationId generation) {
  precond(_initialized);

  const bool is_major = Thread::current() == ZDriver::major();
  const GCCause::Cause cause = is_major ? ZDriver::major()->gc_cause() : ZDriver::minor()->gc_cause();
  const bool is_heap_anti_pressure_gc = cause == GCCause::_z_proactive;
  const bool is_heap_pressure_gc = cause == GCCause::_z_allocation_rate ||
    cause == GCCause::_z_high_usage ||
    cause == GCCause::_z_warmup;

  if (!is_heap_pressure_gc) {
    // If this isn't a GC pressure triggered GC, don't resize or learn anything
    return heap_metrics->_heuristic_max_capacity;
  }

  // Square-root and combined policies: avoid the expensive pressure metric
  // gathering and dispatch directly.  The proactive shrink step is preserved.
  if (ZHeapSizingPolicy == 1 || ZHeapSizingPolicy == 2) {
    if (is_heap_anti_pressure_gc) {
      const size_t selected_capacity = MAX2(size_t(double(heap_metrics->_heuristic_max_capacity) * 0.95),
                                           heap_metrics->_used);
      return clamp(align_down(selected_capacity, ZGranuleSize),
                   heap_metrics->_static_min_capacity,
                   heap_metrics->_current_max_capacity);
    }

    // The Kirisame formula is designed around the old-generation live set (L).
    // Young-generation live bytes are near zero by definition (most young
    // objects die before mark end), which degenerates the formula to the 2 MB
    // floor.  With a 3 GB/s allocation rate and an 8 MB heap the JVM would GC
    // ~375 times per second, completely dominating mutator time.
    //
    // Only run the formula after a major (old-gen) GC, where L is the
    // persistent live data and the result is meaningful.  For young GCs, leave
    // the current heap target unchanged; the next major GC will resize.
    if (generation == ZGenerationId::young) {
      return heap_metrics->_heuristic_max_capacity;
    }

    if (ZHeapSizingPolicy == 1) {
      return compute_heap_size_sqrt(heap_metrics, generation);
    }
    return compute_heap_size_combined(heap_metrics, generation);
  }

  // System memory load
  ZMemoryPressureMetrics mem_metrics = memory_pressure_metrics();

  // System CPU load
  ZCpuPressureMetrics cpu_metrics = cpu_pressure_metrics(generation);

  // Heap size metrics
  const size_t soft_max_capacity = heap_metrics->_soft_max_capacity;
  const size_t current_max_capacity = heap_metrics->_current_max_capacity;
  const size_t heuristic_max_capacity = heap_metrics->_heuristic_max_capacity;
  const size_t capacity = heap_metrics->_capacity;
  const size_t static_min_capacity = heap_metrics->_static_min_capacity;
  const size_t used = heap_metrics->_used;

  if (is_heap_anti_pressure_gc) {
    // The GC is bored. The impact of shrinking should not cost a considerable amount of
    // CPU, or we would not get here.
    const size_t selected_capacity = MAX2(size_t(double(heuristic_max_capacity) * 0.95), used);
    return clamp(align_down(selected_capacity, ZGranuleSize), static_min_capacity, current_max_capacity);
  }

  ZStatCycleStats cycle_stats = ZGeneration::generation(generation)->stat_cycle()->stats();

  const double warmup_time_seconds = 3.0;
  const double warmness = MIN2(os::elapsedTime(), warmup_time_seconds) / warmup_time_seconds;
  const double warmness_squared = warmness * warmness;

  const size_t process_used_memory = os::rss();
  const size_t process_non_heap_memory = process_used_memory > capacity ? process_used_memory - capacity : 0;
  const size_t projected_process_used_memory = heuristic_max_capacity + process_non_heap_memory;

  // Calculate the GC pressure that scales the rest of the heuristics
  ZResourcePressure pressures = compute_pressures(mem_metrics, cpu_metrics, projected_process_used_memory);
  const double gc_intensity = pressures._gc_intensity;
  const double mem_pressure = pressures._mem_pressure;
  const double avg_time_since_last = cpu_metrics._avg_gc_interval;

  // Calculate the heuristic lower bound for the heuristic heap
  const double alloc_rate = heap_metrics->_alloc_rate;
  // Since a GC cycle is obviously round, we can estimate the minimum bytes due to
  // a particular allocation rate and GC intensity by calculating GC intensity * pi.
  const double alloc_rate_scaling = warmness_squared / (gc_intensity * M_PI);
  const size_t heuristic_low = align_down(size_t(double(MAX2(size_t(double(used) * 1.1), size_t(alloc_rate * alloc_rate_scaling))) / mem_pressure), ZGranuleSize);

  const size_t upper_bound = MIN2(soft_max_capacity, current_max_capacity);
  const size_t lower_bound = clamp(heuristic_low, static_min_capacity, upper_bound);

  // When GC intensity is 10, the implication is that we want 25% of the
  // process CPU to be spent on doing GC when the process uses 100% of the
  // available CPU cores.. The ConcGCThreads sizing by default goes up to
  // a maximum of 25% of the available cores. So all ConcGCThreads would
  // be running back to back then.
  const double target_cpu_overhead = gc_intensity / 40.0;

  // Save some breadcrumbs to the director to not use more conc GC threads
  // than we need to run back to back GC at the target GC CPU overhead limit.
  // It is better to let concurrent heap expansion run.
  const double avg_process_time = cpu_metrics._avg_process_time;
  const double avg_process_cpus = avg_process_time / avg_time_since_last;
  const double high_target_workers = avg_process_cpus * target_cpu_overhead;
  uint initial_young_worker_cap = clamp<uint>((uint)ceil(high_target_workers * 1.5), 1, ZYoungGCThreads);
  _initial_young_worker_cap.store_relaxed(initial_young_worker_cap);

  const double upper_cpu_overhead = MAX2(cpu_metrics._avg_total_gc_cpu_overhead, cpu_metrics._generation_gc_cpu_overhead);
  const double upper_cpu_overhead_error = upper_cpu_overhead - target_cpu_overhead;

  const double lower_cpu_overhead = MIN2(cpu_metrics._avg_total_gc_cpu_overhead, cpu_metrics._generation_gc_cpu_overhead);
  const double lower_cpu_overhead_error = lower_cpu_overhead - target_cpu_overhead;

  const double target_gc_interval = compute_target_gc_interval(mem_metrics, gc_intensity);
  const double gc_interval_error = MAX2(target_gc_interval - avg_time_since_last, target_gc_interval - cpu_metrics._avg_gc_interval);

  const double upper_error_signal = MAX2(upper_cpu_overhead_error, gc_interval_error);
  const double lower_error_signal = MAX2(lower_cpu_overhead_error, gc_interval_error);

  const bool is_young = generation == ZGenerationId::young;

  if (is_young) {
    _accumulated_young_gc_time += cpu_metrics._gc_time;
  } else {
    const double young_to_old_gc_time = _accumulated_young_gc_time / (_accumulated_young_gc_time + cycle_stats._last_total_vtime);
    _young_to_old_gc_time.store_relaxed(young_to_old_gc_time);
    _accumulated_young_gc_time = 0.0;
  }

  const double upper_smoothened_error = smoothing_function(upper_error_signal, warmness);
  const double upper_correction_factor = upper_smoothened_error + 0.5;

  const double lower_smoothened_error = smoothing_function(lower_error_signal, warmness);
  const double lower_correction_factor = lower_smoothened_error + 0.5;

  const size_t upper_scaled_capacity = align_up(size_t(double(heuristic_max_capacity) * upper_correction_factor), ZGranuleSize);
  const size_t lower_scaled_capacity = align_up(size_t(double(heuristic_max_capacity) * lower_correction_factor), ZGranuleSize);

  // Instead of increasing the heap by an aggressive amount, reduce the memory
  // availability by said amount as we get closer to the current max capacity.
  // This ensures the heap increase slows down as we approach the max capacity,
  // similar to when a space ship is docking at a station.
  const size_t remaining_capacity = capacity >= current_max_capacity ? 0 : current_max_capacity - capacity;
  const size_t upper_scaled_max_capacity = upper_correction_factor <= 1.0
    ? upper_scaled_capacity
    : align_up(heuristic_max_capacity + size_t(double(remaining_capacity) * (upper_correction_factor - 1.0)), ZGranuleSize);
  const size_t lower_scaled_max_capacity = lower_correction_factor <= 1.0
    ? lower_scaled_capacity
    : align_up(heuristic_max_capacity + size_t(double(remaining_capacity) * (lower_correction_factor - 1.0)), ZGranuleSize);

  const size_t upper_suggested_capacity = MIN2(upper_scaled_capacity, upper_scaled_max_capacity);
  const size_t lower_suggested_capacity = MIN2(lower_scaled_capacity, lower_scaled_max_capacity);

  const size_t upper_bounded_capacity = clamp(upper_suggested_capacity, lower_bound, upper_bound);
  const size_t lower_bounded_capacity = clamp(lower_suggested_capacity, lower_bound, upper_bound);

  // Grow if we experience short term *and* long term pressure on the heap
  const bool should_grow = lower_bounded_capacity > heuristic_max_capacity && upper_bounded_capacity > heuristic_max_capacity;
  // Shrink if we experience short term *and* long term reverse pressure on the heap
  const bool should_shrink = lower_bounded_capacity < heuristic_max_capacity && upper_bounded_capacity < heuristic_max_capacity;

  const double cpu_pressure = pressures._cpu_pressure;
  const double cpu_vs_memory_pressure = pressures._cpu_vs_memory_pressure;
  const double cpu_vs_latency_pressure = pressures._cpu_vs_latency_pressure;

  if (mem_metrics._is_containerized) {
    log_info(gc, load)("Container: System Memory Load: %.1f%%, Process Memory Load: %.1f%%, Heap Memory Load: %.1f%%",
                       double(mem_metrics._container._used_memory) / double(mem_metrics._container._max_memory) * 100.0,
                       double(projected_process_used_memory) / double(mem_metrics._container._max_memory) * 100.0,
                       double(heuristic_max_capacity) / double(mem_metrics._container._max_memory) * 100.0);
  }

  if (cpu_metrics._is_containerized) {
    log_info(gc, load)("Container: System CPU Load: %.1f%%, Process CPU Load: %.1f%%, GC CPU Load: %.1f%%",
                       cpu_metrics._container._avg_system_load * 100.0, cpu_metrics._container._avg_process_load * 100.0,
                       cpu_metrics._avg_total_gc_cpu_overhead * cpu_metrics._container._avg_process_load * 100.0);
  }

  log_info(gc, load)("Machine: System Memory Load: %.1f%%, Process Memory Load: %.1f%%, Heap Memory Load: %.1f%%",
                     double(mem_metrics._machine._used_memory) / double(mem_metrics._machine._max_memory) * 100.0,
                     double(projected_process_used_memory) / double(mem_metrics._machine._max_memory) * 100.0,
                     double(heuristic_max_capacity) / double(mem_metrics._machine._max_memory) * 100.0);

  log_info(gc, load)("Machine: System CPU Load: %.1f%%, Process CPU Load: %.1f%%, GC CPU Load: %.1f%%",
                     cpu_metrics._machine._avg_system_load * 100.0, cpu_metrics._machine._avg_process_load * 100.0,
                     cpu_metrics._avg_total_gc_cpu_overhead * cpu_metrics._machine._avg_process_load * 100.0);

  if (can_adapt()) {
    log_info(gc, heap)("Process GC CPU Overhead: %.1f%%, Target Process GC CPU Overhead: %.1f%%",
                       cpu_metrics._avg_total_gc_cpu_overhead * 100.0, target_cpu_overhead * 100.0);

    log_debug(gc, heap)("System CPU Pressure: %.1f, System Memory Pressure: %.1f",
                        cpu_pressure, mem_pressure);
    log_debug(gc, heap)("System CPU vs Memory Pressure: %.1f, System CPU vs Latency Pressure: %.1f",
                        cpu_vs_memory_pressure, cpu_vs_latency_pressure);
    log_info(gc, heap)("GC Intensity: %.1f, Resource Pressure Scaling: %.1f",
                       gc_intensity, gc_intensity / mem_metrics._unscaled_gc_intensity);

    log_debug(gc, heap)("GC Interval: %.3fs, Target Minimum: %.3fs",
                        avg_time_since_last, target_gc_interval);
    log_debug(gc, heap)("Target heap lower bound: %zuM, upper bound: %zuM",
                        lower_bound / M, upper_bound / M);
    log_debug(gc, heap)("Suggested capacity range: %zuM - %zuM, heuristic capacity: %zuM",
                        lower_suggested_capacity / M, upper_suggested_capacity / M, heuristic_max_capacity / M);
  }

  if (should_grow) {
    const size_t selected_capacity = MIN2(upper_bounded_capacity, lower_bounded_capacity);
    const size_t capacity_resize = selected_capacity - heuristic_max_capacity;

    log_debug(gc, heap)("Updated heuristic max capacity: %zuM (%.3f%%), current capacity: %zuM",
                        selected_capacity / M, double(selected_capacity) / double(heuristic_max_capacity) * 100.0 - 100.0, capacity / M);

    log_info(gc, heap)("Heap Increase %zuM (%.1f%%)", capacity_resize / M, double(capacity_resize) / double(heuristic_max_capacity) * 100.0);

    return selected_capacity;
  } else if (should_shrink) {
    // We want to shrink slower than we grow; by splitting the proposed shrinking into a fraction,
    // we get a slower tail of shrinking, which avoids unnecessary fluctuations up and down.
    const size_t shrinking_fraction = 5;

    const size_t proposed_selected_capacity = MAX2(upper_bounded_capacity, lower_bounded_capacity);
    const size_t capacity_resize = align_up((heuristic_max_capacity - proposed_selected_capacity) / shrinking_fraction, ZGranuleSize);

    const size_t selected_capacity = heuristic_max_capacity - capacity_resize;

    log_debug(gc, heap)("Updated heuristic max capacity: %zuM (%.3f%%), current capacity: %zuM",
                        selected_capacity / M, double(selected_capacity) / double(heuristic_max_capacity) * 100.0 - 100.0, capacity / M);

    log_info(gc, heap)("Heap Decrease %zuM (%.1f%%)", capacity_resize / M, double(capacity_resize) / double(heuristic_max_capacity) * 100.0);

    return selected_capacity;
  }

  return heuristic_max_capacity;
}

uint64_t ZAdaptiveHeap::no_uncommit_delay() {
  return std::numeric_limits<uint64_t>::max();
}

uint64_t ZAdaptiveHeap::urgent_uncommit_delay() {
  return 500;
}

uint64_t ZAdaptiveHeap::critical_uncommit_delay() {
  return 0;
}

static uint64_t system_uncommit_delay(const ZSystemMemoryPressureMetrics& metrics, size_t capacity) {
  const size_t available_memory = metrics._max_memory - metrics._used_memory;

  const double capacity_fraction = clamp(double(capacity) / double(metrics._used_memory), 0.05, 1.0);

  // The remaining memory reserve of the system
  const double available_fraction = double(available_memory) / double(metrics._max_memory);

  // If we are critically low on memory, aggressively free up memory
  if (available_fraction <= metrics._critical_threshold) {
    return ZAdaptiveHeap::critical_uncommit_delay();
  }

  // If we aren't low on memory, disable timer based uncommit; let
  // the GC heuristics guide the heap down instead, as part of the
  // natural control system.
  if (available_fraction > metrics._concerning_threshold) {
    return ZAdaptiveHeap::no_uncommit_delay();
  }

  const uint64_t urgent_delay = ZAdaptiveHeap::urgent_uncommit_delay();

  // If we aren't using a high amount memory, uncommit memory rather slowly
  // and let the GC heuristics do most of the heavy lifting
  if (available_fraction > metrics._high_threshold) {
    // The memory pressure is concerning but not high; gradually siphon the
    // heap to potential other JVMs that may be under more pressure, allowing
    // them to grow.
    // Progression until critical uncommitting starts
    const double progression = calculate_progression(available_fraction, metrics._concerning_threshold, metrics._high_threshold);

    // Scale the uncommit interval by memory urgency, so the pace of uncommitting
    // ramps up as the machine resources gets exhausted.
    return uint64_t((1.0 - progression) * capacity_fraction * (ZUncommitDelay * 1000 - urgent_delay) + urgent_delay);
  }

  // We use a policy where the uncommit delay drops off fairly quickly
  // as the memory pressure gets "high" to let uncommitting react before
  // the next GC, but still without being brutal.
  // When the memory availability becomes critical, more brutal uncommitting
  // will commence.

  // Progression until critical uncommitting starts
  const double progression = calculate_progression(available_fraction, metrics._high_threshold, metrics._critical_threshold);

  // Scale the uncommit interval by memory urgency, so the pace of uncommitting
  // ramps up as the machine resources gets exhausted.
  return uint64_t(urgent_delay * (1.0 - progression));
}

// How long to wait until it is time to uncommit memory. This goes towards
// infinity when there is no concerning memory pressure, then from ZUncommitDelay
// when concerning down to 500 when high, and eventually 0 when critically low.
uint64_t ZAdaptiveHeap::uncommit_delay() {
  precond(_initialized);
  ZMemoryPressureMetrics metrics = memory_pressure_metrics();
  const size_t capacity = ZHeap::heap()->capacity();

  const uint64_t machine_uncommit_delay = system_uncommit_delay(metrics._machine, capacity);

  if (!metrics._is_containerized) {
    return machine_uncommit_delay;
  }

  const uint64_t container_uncommit_delay = system_uncommit_delay(metrics._container, capacity);
  return MIN2(machine_uncommit_delay, container_uncommit_delay);
}

uint64_t ZAdaptiveHeap::soft_ref_delay() {
  precond(_initialized);
  ZStatHeap* const stats = ZGeneration::old()->stat_heap();
  // Young generation should have mostly transient state;
  // consider it as basically free.
  const size_t old_used_reloc_end = stats->used_generation_at_relocate_end();
  const size_t target_capacity = MAX2(ZHeap::heap()->heuristic_max_capacity(), old_used_reloc_end);
  const size_t free_heap = target_capacity - old_used_reloc_end;

  const uint64_t explicit_delay = free_heap / M * uint64_t(SoftRefLRUPolicyMSPerMB);

  if (!can_adapt()) {
    // Use the good old policy we all know and love so much when automatic heap
    // sizing is not in use.
    return explicit_delay;
  }

  // With automatic heap sizing, there is a risk for a feedback loop when the amount
  // free memory decides how long soft references survive. More soft references will
  // lead to the heap growing, hence creating more free memory and suddenly letting
  // soft references live for longer. In order to cut this feedback loop, a more
  // involved policy is used.
  //
  // The more involved strategy scales the delay with the time it would take for the
  // heap to get filled up by old generation allocations multiplied by a scaled
  // variation of SoftRefLRUPolicyMSPerMB. The scaling is more aggressive than linear
  // by computing the nth root of SoftRefLRUPolicyMSPerMB, where n is some memory
  // pressure.

  // Scale the delay by the old generation allocation rate; the faster it fills up,
  // the more rapidly we need to prune soft references
  const double avg_time_since_last = _old_data._gc_times_since_last.avg();
  const size_t old_live = stats->live_at_mark_end();
  const size_t old_used = ZHeap::heap()->used_old();
  const double old_allocated = old_used - old_live;
  const double old_alloc_rate = MAX2(old_allocated / M / avg_time_since_last, 1.0);

  const double time_to_old_oom = double(free_heap) / M / old_alloc_rate;

  const double free_ratio = double(target_capacity) / double(free_heap);

  const double mem_pressure = free_ratio;

  // No point to clear more soft references due to external memory pressure if the
  // Scale the SoftRefLRUPolicyMSPerMB as an nth root where n is the memory pressure.
  // The reason for using the nth root is that it might not necessarily be that
  // linearly decreasing the interval with memory pressure yields linearly more
  // soft references being cleared. It rather depends on the access frequency.
  // If they get accessed very frequently, then it's likely that no soft reference
  // get cleared at all, until the interval is made *very* small. Therefore, the
  // more aggressive nth root is used.
  const uint64_t scaled_interval = uint64_t(pow(SoftRefLRUPolicyMSPerMB, 1.0 / mem_pressure));

  // Compute the potentially more aggressive delay that cuts the feedback loop.
  const uint64_t implicit_delay = uint64_t(time_to_old_oom * double(scaled_interval));

  // If the new policy yields earlier cut off points, then use that. Otherwise,
  // we still use the more relaxed policy to cut off soft references when they
  // have not been used for unreasonably long. While we could keep them around
  // forever, it might also be a bit pointless.
  const uint64_t delay = MIN2(implicit_delay, explicit_delay);

  log_info(gc, ref)("Soft ref timeout: %.3fs", double(delay) / 1000);

  LogTarget(Debug, gc, ref) lt;
  if (lt.is_enabled()) {
    LogStream ls(lt);

    ls.print_cr("Soft ref time to old generation OOM: %.3fs", time_to_old_oom);
    ls.print_cr("Soft ref explicit timeout: %.3fs", double(explicit_delay) / 1000);
    ls.print_cr("Soft ref implicit timeout: %.3fs", double(implicit_delay) / 1000);
    ls.print_cr("Soft ref memory pressure: %.3f", mem_pressure);
  }

  return delay;
}

void ZAdaptiveHeap::print() {
  precond(_initialized);
  const char* status;
  if (!can_adapt()) {
    status = "Fixed";
  } else {
    if (explicit_max_capacity() || MinHeapSize != DefaultMinHeapSize) {
      status = "Automatic (Explicit Boundaries)";
    } else {
      status = "Automatic (Implicit Boundaries)";
    }
  }
  log_info_p(gc, init)("Heap Sizing: %s", status);
}
