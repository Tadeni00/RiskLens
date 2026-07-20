"""
FraudTrap — Benchmark Script
Standalone performance benchmark without Locust.
Usage: python tests/load/benchmark.py --host http://localhost:8000 --duration 60 --concurrency 50
"""
import asyncio
import aiohttp
import argparse
import json
import random
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List
import statistics


@dataclass
class BenchmarkResult:
    total_requests: int
    successful: int
    failed: int
    total_time: float
    latencies: List[float]
    errors: List[str]
    
    @property
    def rps(self) -> float:
        return self.total_requests / self.total_time if self.total_time > 0 else 0
    
    @property
    def success_rate(self) -> float:
        return self.successful / self.total_requests if self.total_requests > 0 else 0
    
    @property
    def p50(self) -> float:
        return self._percentile(50)
    
    @property
    def p95(self) -> float:
        return self._percentile(95)
    
    @property
    def p99(self) -> float:
        return self._percentile(99)
    
    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0
    
    def _percentile(self, p: float) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * p / 100)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]


class BenchmarkRunner:
    """Async benchmark runner for FraudTrap API."""
    
    def __init__(
        self,
        host: str,
        concurrency: int = 50,
        duration_seconds: int = 60,
        ramp_up_seconds: int = 10,
    ):
        self.host = host.rstrip("/")
        self.concurrency = concurrency
        self.duration_seconds = duration_seconds
        self.ramp_up_seconds = ramp_up_seconds
        self.results = []
        self.errors = []
        self._stop = False
        self.start_time = None
    
    def _random_transaction(self) -> dict:
        """Generate a realistic transaction payload."""
        return {
            "tenant_id": random.choice(["bank_ng_gtb", "bank_ke_equity", "fintech_za_yoco"]),
            "account_id": f"acct_{random.randint(1, 10000)}",
            "amount": round(random.lognormvariate(9, 1.5), 2),
            "currency": "NGN",
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "transaction_type": random.choice(["PAYMENT", "TRANSFER", "WITHDRAWAL", "TOP_UP", "REFUND"]),
            "channel": random.choice(["MOBILE", "WEB", "API", "POS", "ATM", "USSD"]),
            "device_id": f"dev_{random.randint(1, 5000)}",
            "merchant_id": f"merch_{random.randint(1, 2000)}" if random.random() > 0.2 else None,
            "country_code": "NG" if random.random() > 0.1 else "US",
        }
    
    async def _score_single(self, session: aiohttp.ClientSession) -> tuple[float, bool, str]:
        """Score a single transaction."""
        payload = self._random_transaction()
        start = time.perf_counter()
        try:
            async with session.post(f"{self.host}/v1/score", json=payload) as response:
                latency = (time.perf_counter() - start) * 1000
                if response.status == 200:
                    await response.json()  # Validate JSON
                    return latency, True, ""
                else:
                    text = await response.text()
                    return latency, False, f"HTTP {response.status}: {text[:100]}"
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return latency, False, str(e)
    
    async def _batch_score(self, session: aiohttp.ClientSession, batch_size: int = 50) -> tuple[float, bool, str]:
        """Score a batch of transactions."""
        payload = [self._random_transaction() for _ in range(50)]
        start = time.perf_counter()
        try:
            async with session.post(f"{self.host}/v1/score/batch", json=payload) as response:
                latency = (time.perf_counter() - start) * 1000
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, list) and len(data) == 50:
                        return latency, True, ""
                    return latency, False, f"Invalid batch response: {type(data)}"
                else:
                    return latency, False, f"HTTP {response.status}"
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return latency, False, str(e)
    
    async def _health_check(self, session: aiohttp.ClientSession) -> tuple[float, bool]:
        """Health check endpoint."""
        start = time.perf_counter()
        try:
            async with session.get(f"{self.host}/health") as response:
                latency = (time.perf_counter() - start) * 1000
                return latency, response.status == 200
        except Exception:
            return (time.perf_counter() - start) * 1000, False
    
    async def _recent_scores(self, session: aiohttp.ClientSession) -> tuple[float, bool]:
        """Get recent scores."""
        start = time.perf_counter()
        try:
            async with session.get(f"{self.host}/v1/recent?limit=100") as response:
                latency = (time.perf_counter() - start) * 1000
                if response.status == 200:
                    data = await response.json()
                    return latency, "items" in data
                return latency, False
        except Exception:
            return (time.perf_counter() - start) * 1000, False
    
    async def _worker(self, worker_id: int, session: aiohttp.ClientSession, stop_event: asyncio.Event):
        """Worker coroutine that sends requests until stop_event is set."""
        request_count = 0
        error_count = 0
        
        while not stop_event.is_set():
            # Mix of operations (weighted like real traffic)
            r = random.random()
            if r < 0.85:
                # Score single
                latency, success, error = await self._score_single(self.session)
            elif r < 0.93:
                # Batch
                latency, success, error = await self._batch_score(self.session)
            elif r < 0.96:
                # Health
                latency, success = await self._health_check(self.session)
                error = ""
            else:
                # Recent
                latency, success = await self._recent_scores(self.session)
                error = ""
            
            if success:
                self.results.append(latency)
            else:
                self.errors.append(error)
                error_count += 1
            
            request_count += 1
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(random.uniform(0.001, 0.01))
    
    async def run(self) -> dict:
        """Run the benchmark."""
        print(f"Starting benchmark: {self.host}")
        print(f"  Concurrency: {self.concurrency}")
        print(f"  Duration: {self.duration_seconds}s")
        print(f"  Ramp-up: {self.ramp_up_seconds}s")
        print()
        
        self.results = []
        self.errors = []
        self.start_time = time.time()
        
        # Create session
        connector = aiohttp.TCPConnector(
            limit=self.concurrency * 2,
            limit_per_host=self.concurrency,
            ttl_dns_cache=300,
        )
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            self.session = session
            
            # Health check first
            print("Running health check...")
            for _ in range(3):
                latency, ok = await self._health_check(session)
                if ok:
                    print(f"  Health check OK ({latency:.1f}ms)")
                    break
                await asyncio.sleep(1)
            else:
                print("  WARNING: Health check failed!")
            
            print("Starting benchmark...")
            
            stop_event = asyncio.Event()
            workers = []
            
            # Ramp up workers gradually
            for i in range(self.concurrency):
                if self.ramp_up_seconds > 0:
                    await asyncio.sleep(self.ramp_up_seconds / self.concurrency)
                workers.append(asyncio.create_task(
                    self._worker(i, session, asyncio.Event())
                ))
            
            # Wait for duration
            await asyncio.sleep(self.duration_seconds)
            self._stop = True
            
            # Wait for workers to finish
            await asyncio.gather(*workers, return_exceptions=True)
            
            end_time = time.time()
            total_time = end_time - self.start_time
            
            # Compile results
            successful = len(self.results)
            failed = len(self.errors)
            total = successful + failed
            
            result = {
                "host": self.host,
                "concurrency": self.concurrency,
                "duration_seconds": total_time,
                "total_requests": total,
                "successful": successful,
                "failed": failed,
                "success_rate": successful / max(total, 1) * 100,
                "rps": total / total_time if total_time > 0 else 0,
                "avg_latency_ms": statistics.mean(self.results) if self.results else 0,
                "p50_latency_ms": self._percentile(self.results, 50),
                "p95_latency_ms": self._percentile(self.results, 95),
                "p99_latency_ms": self._percentile(self.results, 99),
                "max_latency_ms": max(self.results) if self.results else 0,
                "error_rate": len(self.errors) / max(total, 1) * 100,
                "unique_errors": list(set(self.errors))[:10],
            }
            
            return result
    
    @staticmethod
    def _percentile(data: list, p: float) -> float:
        if not data:
            return 0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]


async def run_benchmark(
    host: str,
    concurrency: int,
    duration: int,
    ramp_up: int,
    output: str = None,
):
    """Run benchmark and return results."""
    runner = BenchmarkRunner(
        host=host,
        concurrency=concurrency,
        duration_seconds=duration,
        ramp_up_seconds=ramp_up,
    )
    
    results = await runner.run()
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Host:              {results['host']}")
    print(f"Concurrency:       {results['concurrency']}")
    print(f"Duration:          {results['duration_seconds']:.1f}s")
    print(f"Total Requests:    {results['total_requests']:,}")
    print(f"Successful:        {results['successful']:,} ({results['success_rate']:.2f}%)")
    print(f"Failed:            {results['failed']:,} ({results['error_rate']:.2f}%)")
    print(f"RPS:               {results['rps']:.2f}")
    print(f"Avg Latency:       {results['avg_latency_ms']:.2f}ms")
    print(f"P50 Latency:       {results['p50_latency_ms']:.2f}ms")
    print(f"P95 Latency:       {results['p95_latency_ms']:.2f}ms")
    print(f"P99 Latency:       {results['p99_latency_ms']:.2f}ms")
    print(f"Max Latency:       {results['max_latency_ms']:.2f}ms")
    
    if results['unique_errors']:
        print(f"\nErrors ({len(results['unique_errors'])}):")
        for err in results['unique_errors']:
            print(f"  - {err}")
    
    # Pass/fail
    passed = (
        results['p95_latency_ms'] <= 90 and
        results['error_rate'] <= 1.0 and
        results['success_rate'] >= 99.0
    )
    print(f"\n{'PASSED' if passed else 'FAILED'}: P95<={90}ms, ErrorRate<=1%, Success>=99%")
    
    # Save output
    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="FraudTrap API Benchmark")
    parser.add_argument("--host", default="http://localhost:8000", help="API host URL")
    parser.add_argument("-c", "--concurrency", type=int, default=50, help="Concurrent workers")
    parser.add_argument("-d", "--duration", type=int, default=60, help="Test duration (seconds)")
    parser.add_argument("-r", "--ramp-up", type=int, default=10, help="Ramp-up time (seconds)")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("--scenario", choices=["baseline", "peak", "spike", "soak"], help="Predefined scenario")
    args = parser.parse_args()
    
    # Scenario presets
    scenarios = {
        "baseline": {"concurrency": 50, "duration": 600, "ramp_up": 30},
        "peak": {"concurrency": 250, "duration": 300, "ramp_up": 60},
        "spike": {"concurrency": 500, "duration": 120, "ramp_up": 10},
        "soak": {"concurrency": 100, "duration": 3600, "ramp_up": 60},
    }
    
    if args.scenario:
        params = scenarios[args.scenario]
        print(f"Running '{args.scenario}' scenario: {params}")
        args.concurrency = params["concurrency"]
        args.duration = params["duration"]
        args.ramp_up = params["ramp_up"]
    
    asyncio.run(run_benchmark(
        host=args.host,
        concurrency=args.concurrency,
        duration=args.duration,
        ramp_up=args.ramp_up,
        output=args.output,
    ))


if __name__ == "__main__":
    main()