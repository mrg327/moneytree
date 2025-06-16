#!/usr/bin/env python3
"""
Run all MoneyTree pipeline tests to identify issues and bottlenecks.

This script runs comprehensive tests for each component of the pipeline:
- Wikipedia content fetching
- Content generation (LLM and rule-based)
- Speech synthesis (ChatTTS and Coqui)
- Video processing and rendering
- Full pipeline integration

Usage: python tests/run_all_tests.py
"""

import sys
import time
from pathlib import Path

def run_test_module(module_name, test_file):
    """Run a specific test module and capture results."""
    print(f"\\n{'='*60}")
    print(f"🧪 RUNNING: {module_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Import and run the test module
        exec(open(test_file).read())
        
        duration = time.time() - start_time
        print(f"\\n✅ {module_name} completed in {duration:.2f}s")
        return True, duration
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"\\n❌ {module_name} failed after {duration:.2f}s: {e}")
        return False, duration

def main():
    """Run all pipeline tests."""
    print("🌳 MoneyTree Pipeline Test Suite")
    print("="*60)
    print("This will test each component of the video generation pipeline")
    print("to identify bottlenecks and issues.\\n")
    
    # Define test modules
    test_modules = [
        ("Wikipedia Crawler", "tests/wiki/test_wikipedia_crawler.py"),
        ("Content Generation", "tests/llm/test_content_generation.py"), 
        ("Speech Generation", "tests/tts/test_speech_generation.py"),
        ("Video Processing", "tests/video/test_video_processing.py"),
        ("Full Pipeline Integration", "tests/integration/test_full_pipeline.py")
    ]
    
    results = []
    total_start_time = time.time()
    
    # Run each test module
    for module_name, test_file in test_modules:
        if not Path(test_file).exists():
            print(f"⚠️ Test file not found: {test_file}")
            results.append((module_name, False, 0))
            continue
        
        success, duration = run_test_module(module_name, test_file)
        results.append((module_name, success, duration))
    
    total_duration = time.time() - total_start_time
    
    # Print summary
    print(f"\\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for module_name, success, duration in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {module_name:<30} ({duration:.2f}s)")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\\n📈 OVERALL RESULTS:")
    print(f"   Total time: {total_duration:.2f}s")
    print(f"   Tests passed: {passed}")
    print(f"   Tests failed: {failed}")
    print(f"   Success rate: {(passed/(passed+failed)*100):.1f}%" if (passed+failed) > 0 else "N/A")
    
    # Recommendations based on results
    print(f"\\n💡 RECOMMENDATIONS:")
    
    if failed == 0:
        print("   🎉 All tests passed! The pipeline should work correctly.")
    else:
        print("   🔧 Focus debugging efforts on failed components:")
        
        for module_name, success, duration in results:
            if not success:
                if "Speech Generation" in module_name:
                    print("      - Check ChatTTS model loading and audio file permissions")
                elif "Video Processing" in module_name:
                    print("      - Check MoviePy dependencies and template video files")
                elif "Wikipedia" in module_name:
                    print("      - Check internet connection and Wikipedia API access")
                elif "Content Generation" in module_name:
                    print("      - Check LLM availability (Ollama) and rule-based fallbacks")
                elif "Integration" in module_name:
                    print("      - Check that individual components work before integration")
    
    # Performance analysis
    slow_tests = [(name, dur) for name, success, dur in results if success and dur > 5.0]
    if slow_tests:
        print(f"\\n⚡ PERFORMANCE BOTTLENECKS:")
        for name, duration in slow_tests:
            print(f"   - {name}: {duration:.2f}s (consider optimization)")
    
    # Exit with appropriate code
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)