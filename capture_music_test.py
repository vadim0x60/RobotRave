#!/usr/bin/env python3
"""
Capture real music test - records servo outputs while mic captures music.
"""

import asyncio
import json
import time
import numpy as np
from collections import defaultdict
import sounddevice as sd
import websockets

SERVER = "ws://132.145.180.105:8765"
SAMPLE_RATE = 30720
CHUNK_SIZE = 3072

async def capture_and_record(duration_sec, label):
    """Capture mic audio, stream to server, record servo outputs."""

    print(f"\n{'='*60}")
    print(f"Recording: {label} ({duration_sec}s)")
    print(f"{'='*60}")

    servo_history = defaultdict(list)
    timestamps = []
    loop = asyncio.get_running_loop()
    audio_queue = asyncio.Queue()

    def audio_callback(indata, frames, time_info, status):
        audio = indata[:, 0].astype(np.float32)
        loop.call_soon_threadsafe(audio_queue.put_nowait, audio.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    )

    try:
        async with websockets.connect(SERVER, max_size=10*1024*1024) as ws:
            print("Connected to server, listening...")
            stream.start()
            start_time = time.time()
            first_response = None

            async def send_audio():
                while time.time() - start_time < duration_sec + 2:
                    try:
                        chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
                        await ws.send(json.dumps({
                            'type': 'audio',
                            'audio': chunk.tolist()
                        }))
                    except asyncio.TimeoutError:
                        pass

            async def receive_servos():
                nonlocal first_response
                while time.time() - start_time < duration_sec + 5:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        data = json.loads(msg)
                        if data['type'] == 'servos':
                            now = time.time() - start_time
                            if first_response is None:
                                first_response = now
                                print(f"First response at {now:.1f}s")

                            timestamps.append(now)
                            for sid, val in data['servos'].items():
                                servo_history[sid].append(val)

                            # Progress indicator
                            if len(timestamps) % 20 == 0:
                                print(f"  {now:.0f}s - {len(timestamps)} responses")
                    except asyncio.TimeoutError:
                        if time.time() - start_time > duration_sec + 3:
                            break

            await asyncio.gather(send_audio(), receive_servos())

    finally:
        stream.stop()
        stream.close()

    # Calculate stats
    results = {'label': label, 'responses': len(timestamps), 'servos': {}}

    print(f"\nResults for {label}:")
    for sid in sorted(servo_history.keys()):
        vals = np.array(servo_history[sid])
        stats = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'range': float(np.ptp(vals)),
            'values': vals.tolist()  # Keep raw values for analysis
        }
        results['servos'][sid] = stats
        print(f"  Servo {sid}: mean={stats['mean']:.0f}, std={stats['std']:.1f}, range={stats['range']:.0f}")

    return results

def analyze_results(all_results):
    """Compare results across different music types."""

    print("\n" + "="*60)
    print("COMPARISON ANALYSIS")
    print("="*60)

    # Movement scores
    print("\nTotal Movement (sum of std devs):")
    for r in all_results:
        total = sum(s['std'] for s in r['servos'].values())
        bar = "█" * int(total / 10)
        print(f"  {r['label']:<20}: {total:>6.1f} {bar}")

    # Per-servo comparison
    print("\nPer-Servo Standard Deviation:")
    servo_ids = sorted(all_results[0]['servos'].keys())
    header = f"{'Genre':<20}" + "".join(f"  S{s:<5}" for s in servo_ids)
    print(header)
    print("-" * len(header))

    for r in all_results:
        row = f"{r['label']:<20}"
        for sid in servo_ids:
            row += f"  {r['servos'][sid]['std']:>5.1f}"
        print(row)

    # Tempo analysis (rate of change)
    print("\nMovement Rate (avg change between frames):")
    for r in all_results:
        total_change = 0
        for sid, stats in r['servos'].items():
            vals = np.array(stats['values'])
            if len(vals) > 1:
                changes = np.abs(np.diff(vals))
                total_change += np.mean(changes)
        print(f"  {r['label']:<20}: {total_change:.1f} units/frame")

async def main():
    all_results = []

    # Test 1: AC/DC
    input("\n>>> Press ENTER, then start playing AC/DC...")
    print("GO! Playing AC/DC for 30 seconds...")
    result = await capture_and_record(30, "AC/DC (Rock)")
    all_results.append(result)

    # Test 2: Waltz
    input("\n>>> Press ENTER, then start playing a Waltz...")
    print("GO! Playing Waltz for 30 seconds...")
    result = await capture_and_record(30, "Waltz (Classical)")
    all_results.append(result)

    # Test 3: Infected Mushroom
    input("\n>>> Press ENTER, then start playing Infected Mushroom...")
    print("GO! Playing Infected Mushroom for 30 seconds...")
    result = await capture_and_record(30, "Infected Mushroom (Psy)")
    all_results.append(result)

    # Analyze
    analyze_results(all_results)

    # Save raw data
    import json as json_module
    with open('music_test_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON
        save_data = []
        for r in all_results:
            save_r = {'label': r['label'], 'responses': r['responses'], 'servos': {}}
            for sid, stats in r['servos'].items():
                save_r['servos'][sid] = {k: v for k, v in stats.items() if k != 'values'}
            save_data.append(save_r)
        json_module.dump(save_data, f, indent=2)
    print("\nRaw data saved to music_test_results.json")

if __name__ == '__main__':
    asyncio.run(main())
