/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

#ifndef SHARE_GC_Z_ZADAPTIVEHEAP_HPP
#define SHARE_GC_Z_ZADAPTIVEHEAP_HPP

#include "gc/z/zGenerationId.hpp"
#include "gc/z/zStat.hpp"
#include "memory/allocation.hpp"
#include "utilities/numberSeq.hpp"

struct ZHeapResizeMetrics {
  const size_t _soft_max_capacity;
  const size_t _current_max_capacity;
  const size_t _heuristic_max_capacity;
  const size_t _static_min_capacity;
  const size_t _capacity;
  const size_t _used;
  const double _alloc_rate;
};

struct ZSystemMemoryPressureMetrics {
  const physical_memory_size_type _used_memory;
  const physical_memory_size_type _max_memory;

  const double _concerning_threshold;
  const double _high_threshold;
  const double _critical_threshold;
};

struct ZMemoryPressureMetrics {
  const double _unscaled_gc_intensity;
  const bool _is_containerized;
  const ZSystemMemoryPressureMetrics _machine;
  const ZSystemMemoryPressureMetrics _container;
};

struct ZSystemCpuPressureMetrics {
  const double _avg_process_load;
  const double _avg_system_load;
};

struct ZCpuPressureMetrics {
  const bool _is_containerized;
  const double _generation_gc_cpu_overhead;
  const double _avg_generation_gc_cpu_overhead;
  const double _avg_total_gc_cpu_overhead;
  const double _avg_gc_interval;
  const double _avg_process_time;
  const double _gc_time;
  const ZSystemCpuPressureMetrics _machine;
  const ZSystemCpuPressureMetrics _container;
};

struct ZResourcePressure {
  const double _gc_intensity;
  const double _cpu_pressure;
  const double _mem_pressure;
  const double _cpu_vs_memory_pressure;
  const double _cpu_vs_latency_pressure;
};

class ZAdaptiveHeap : public AllStatic {
  friend class ZAdaptiveHeapTest;
private:
  static bool _explicit_max_capacity;
  static bool _can_adapt;
  static bool _initialized;
  static TruncatedSeq _gc_intensities;

  struct ZGenerationOverhead {
    double       _last_machine_system_time;
    double       _last_container_system_time;
    double       _last_process_time;
    double       _last_time;
    TruncatedSeq _process_times;
    TruncatedSeq _machine_system_times;
    TruncatedSeq _container_system_times;
    TruncatedSeq _gc_times;
    TruncatedSeq _gc_times_since_last;

    ZGenerationOverhead()
      : _last_machine_system_time(),
        _last_container_system_time(),
        _last_process_time(),
        _process_times(),
        _machine_system_times(),
        _container_system_times(),
        _gc_times(),
        _gc_times_since_last() {}
  };

  static Atomic<double> _young_to_old_gc_time;
  static double _accumulated_young_gc_time;
  static ZGenerationOverhead _young_data;
  static ZGenerationOverhead _old_data;
  static Atomic<uint> _initial_young_worker_cap;

  // Per-generation EWMA state for the square-root heap sizing policy.
  // Tracks smoothed live bytes, GC CPU time, and allocation rate following
  // the smoothing parameters from the "Optimal Heap Limits" paper.
  struct ZSqrtSizerState {
    double _avg_live_bytes;
    double _avg_gc_cpu_time;
    double _avg_alloc_rate;
    // Peak-hold wall time and allocation rate, each with slow exponential
    // decay. EWMAs of these signals are dominated by fast uncontested cycles,
    // which hides the burst conditions E_stall must react to: brief windows
    // where the mutator allocates at 5–10 GB/s while GC wall time stretches
    // under CPU contention. Peak-hold captures the worst observed value and
    // decays it over ~40 cycles so E_stall stays elevated across the burst
    // window instead of collapsing between spikes.
    double _peak_gc_wall_time;
    double _peak_alloc_rate;

    ZSqrtSizerState()
      : _avg_live_bytes(0.0),
        _avg_gc_cpu_time(0.0),
        _avg_alloc_rate(0.0),
        _peak_gc_wall_time(0.0),
        _peak_alloc_rate(0.0) {}
  };

  static ZSqrtSizerState _sqrt_young_state;
  static ZSqrtSizerState _sqrt_old_state;

  static ZCpuPressureMetrics cpu_pressure_metrics(ZGenerationId generation);

  static ZResourcePressure compute_pressures(const ZMemoryPressureMetrics& mem_metrics,
                                             const ZCpuPressureMetrics& cpu_metrics,
                                             size_t projected_process_used_memory);
  static double compute_memory_pressure(const ZMemoryPressureMetrics& metrics);

  static void   update_sqrt_state(ZSqrtSizerState& state, size_t live, double gc_cpu_time, double alloc_rate, double gc_wall_time);
  static size_t compute_heap_size_sqrt(ZHeapResizeMetrics* metrics, ZGenerationId generation);
  static size_t compute_heap_size_combined(ZHeapResizeMetrics* metrics, ZGenerationId generation);

public:
  static void initialize(bool explicit_max_heap_size, bool can_adapt);

  static size_t compute_heap_size(ZHeapResizeMetrics* metrics, ZGenerationId generation);
  static double young_to_old_gc_time();
  static uint initial_young_worker_cap();

  // Uncommit support
  static uint64_t no_uncommit_delay();
  static uint64_t urgent_uncommit_delay();
  static uint64_t critical_uncommit_delay();
  static uint64_t uncommit_delay();

  static uint64_t soft_ref_delay();

  // Memory pressure
  static ZMemoryPressureMetrics memory_pressure_metrics();
  static bool is_memory_pressure_concerning(const ZMemoryPressureMetrics& metrics);
  static bool is_memory_pressure_high(const ZMemoryPressureMetrics& metrics);
  static bool is_memory_pressure_critical(const ZMemoryPressureMetrics& metrics);

  // How adaptive are we?
  static bool explicit_max_capacity();
  static bool can_adapt();

  static constexpr double DefaultMaxRAMPercentage = 100.;
  static constexpr size_t DefaultMinHeapSize = 2 * M;

  static void print();
};

#endif // SHARE_GC_Z_ZADAPTIVEHEAP_HPP
