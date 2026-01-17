#!/usr/bin/env python3
"""
Test whether FACT server responds differently to different audio inputs.

Plays test audio files and records servo outputs to compare.

Usage:
    python test_music_response.py --server ws://132.145.180.105:8765
"""

import asyncio
import json
import time
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

import websockets

# Test with generated audio (no files needed)
def generate_silence(duration_sec, sample_rate=30720):
    """Generate silence."""
    return np.zeros(int(duration_sec * sample_rate), dtype=np.float32)

def generate_sine(duration_sec, freq=440, sample_rate=30720):
    """Generate a pure sine wave."""
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)

def generate_beats(duration_sec, bpm=120, sample_rate=30720):
    """Generate rhythmic beats (kick drum simulation)."""
    samples = int(duration_sec * sample_rate)
    audio = np.zeros(samples, dtype=np.float32)
    beat_interval = int(sample_rate * 60 / bpm)

    for i in range(0, samples, beat_interval):
        # Simple kick: exponential decay
        kick_len = min(int(sample_rate * 0.1), samples - i)
        t = np.arange(kick_len) / sample_rate
        kick = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 20)
        audio[i:i+kick_len] += kick.astype(np.float32)

    return np.clip(audio, -1, 1).astype(np.float32)

def generate_noise(duration_sec, sample_rate=30720):
    """Generate white noise."""
    return (np.random.randn(int(duration_sec * sample_rate)) * 0.3).astype(np.float32)

def generate_sweep(duration_sec, sample_rate=30720):
    """Generate frequency sweep (low to high)."""
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate), dtype=np.float32)
    freq = 100 + (2000 - 100) * t / duration_sec  # 100Hz to 2000Hz
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    return (0.5 * np.sin(phase)).astype(np.float32)


async def test_audio(server_url, audio_data, audio_name, chunk_size=3072):
    """Send audio to server and collect servo responses."""

    print(f"\nTesting: {audio_name}")
    print("-" * 40)

    servo_history = defaultdict(list)
    timestamps = []

    try:
        async with websockets.connect(server_url, max_size=10*1024*1024) as ws:
            # Send all audio in chunks
            start_time = time.time()
            chunk_idx = 0

            async def send_audio():
                nonlocal chunk_idx
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i+chunk_size]
                    await ws.send(json.dumps({
                        'type': 'audio',
                        'audio': chunk.tolist(),
                        'timestamp': time.time()
                    }))
                    chunk_idx += 1
                    await asyncio.sleep(chunk_size / 30720)  # Simulate real-time

            async def receive_servos():
                first_response = None
                while True:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        data = json.loads(message)

                        if data['type'] == 'servos':
                            now = time.time()
                            if first_response is None:
                                first_response = now - start_time
                                print(f"  First response after: {first_response:.2f}s")

                            timestamps.append(now)
                            for servo_id, value in data['servos'].items():
                                servo_history[servo_id].append(value)
                    except asyncio.TimeoutError:
                        break

            # Run both concurrently
            await asyncio.gather(
                send_audio(),
                receive_servos()
            )

    except Exception as e:
        print(f"  Error: {e}")
        return None

    # Analyze results
    if not servo_history:
        print("  No servo data received!")
        return None

    results = {
        'name': audio_name,
        'num_responses': len(timestamps),
        'duration': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
        'servos': {}
    }

    print(f"  Responses: {len(timestamps)}")

    for servo_id, values in sorted(servo_history.items()):
        values = np.array(values)
        stats = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'range': float(np.max(values) - np.min(values)),
        }
        results['servos'][servo_id] = stats
        print(f"  Servo {servo_id}: mean={stats['mean']:.0f}, std={stats['std']:.1f}, range={stats['range']:.0f}")

    return results


async def main(args):
    print("=" * 60)
    print("FACT Music Response Test")
    print("=" * 60)
    print(f"Server: {args.server}")

    duration = args.duration

    # Generate test audio
    test_cases = [
        ("Silence", generate_silence(duration)),
        ("440Hz Sine", generate_sine(duration, freq=440)),
        ("Bass Beats 120BPM", generate_beats(duration, bpm=120)),
        ("Fast Beats 180BPM", generate_beats(duration, bpm=180)),
        ("White Noise", generate_noise(duration)),
        ("Frequency Sweep", generate_sweep(duration)),
    ]

    all_results = []

    for name, audio in test_cases:
        result = await test_audio(args.server, audio, name)
        if result:
            all_results.append(result)
        await asyncio.sleep(1)  # Brief pause between tests

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY: Servo Movement by Audio Type")
    print("=" * 60)

    if all_results:
        # Get all servo IDs
        all_servo_ids = set()
        for r in all_results:
            all_servo_ids.update(r['servos'].keys())

        # Print comparison table
        print(f"\n{'Audio Type':<20} | " + " | ".join(f"S{sid} std" for sid in sorted(all_servo_ids)))
        print("-" * 80)

        for r in all_results:
            row = f"{r['name']:<20} | "
            for sid in sorted(all_servo_ids):
                if sid in r['servos']:
                    row += f"{r['servos'][sid]['std']:>7.1f} | "
                else:
                    row += "    N/A | "
            print(row)

        # Calculate total movement score
        print("\nTotal Movement Score (sum of all servo std devs):")
        for r in all_results:
            total_std = sum(s['std'] for s in r['servos'].values())
            bar = "█" * int(total_std / 5)
            print(f"  {r['name']:<20}: {total_std:>6.1f} {bar}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test FACT music response')
    parser.add_argument('--server', type=str, default='ws://132.145.180.105:8765')
    parser.add_argument('--duration', type=float, default=15.0,
                       help='Duration of each test audio in seconds')
    args = parser.parse_args()

    asyncio.run(main(args))
