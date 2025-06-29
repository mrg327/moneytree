#!/usr/bin/env python3
"""
Demo showcasing YouTube download capabilities for MoneyTree.

Downloads YouTube videos and audio for use as templates and background music.
"""

import sys
import argparse
from pathlib import Path

from lib.download.youtube import YouTubeDownloader, DownloadConfig, get_recommended_channels


def main():
    """Run the YouTube download demo."""
    parser = argparse.ArgumentParser(description='MoneyTree: YouTube Content Downloader')
    parser.add_argument('url', nargs='?', help='YouTube URL to download')
    parser.add_argument('--type', choices=['video', 'audio', 'info'], default='info',
                       help='Download type')
    parser.add_argument('--quality', help='Quality setting (144p, 240p, 360p, 480p, 720p, 1080p, 1440p, 2160p, 4k, 8k, high, highest, best for video; 64, 128, 192, 256, 320, best for audio)')
    parser.add_argument('--output', help='Custom output filename (without extension)')
    parser.add_argument('--format', help='Output format (mp4, webm for video; mp3, m4a for audio)')
    parser.add_argument('--output-dir', default='downloads', help='Output directory')
    parser.add_argument('--list-downloads', action='store_true', help='List previously downloaded files')
    parser.add_argument('--cleanup', type=int, metavar='DAYS', 
                       help='Clean up downloads older than DAYS')
    parser.add_argument('--recommended', action='store_true', 
                       help='Show recommended educational channels')
    parser.add_argument('--codec', choices=['h264', 'vp9', 'av1', 'any'], default='any',
                       help='Preferred video codec')
    parser.add_argument('--60fps', action='store_true',
                       help='Prefer 60fps when available')
    parser.add_argument('--show-formats', action='store_true',
                       help='Show available formats before downloading')
    parser.add_argument('--test-formats', action='store_true',
                       help='Test different quality settings on a URL (no download)')
    
    args = parser.parse_args()
    
    # Show recommended channels
    if args.recommended:
        print("🎬 Recommended Educational YouTube Channels:")
        print("=" * 60)
        for channel in get_recommended_channels():
            print(f"📺 {channel['name']}")
            print(f"   Type: {channel['type']}")
            print(f"   URL: {channel['url']}")
            print(f"   Description: {channel['description']}")
            print()
        return
    
    # Configure downloader
    config = DownloadConfig(output_dir=args.output_dir)
    if args.quality:
        if args.type == 'video':
            config.video_quality = args.quality
        else:
            config.audio_quality = args.quality
    if args.format:
        if args.type == 'video':
            config.output_format = args.format
        else:
            config.audio_format = args.format
    
    # Set advanced video options
    config.codec_preference = args.codec
    config.prefer_60fps = getattr(args, '60fps', False)
    config.show_available_formats = getattr(args, 'show_formats', False)
    
    downloader = YouTubeDownloader(config)
    
    # List downloads
    if args.list_downloads:
        print("📁 Downloaded Files:")
        print("=" * 50)
        downloads = downloader.get_download_history()
        
        if not downloads:
            print("No downloads found.")
            return
        
        for download in downloads:
            size_mb = download['size'] / (1024 * 1024)
            print(f"📄 {download['filename']}")
            print(f"   Type: {download['type'].title()}")
            print(f"   Category: {download['category']}")
            print(f"   Size: {size_mb:.1f} MB")
            print(f"   Path: {download['path']}")
            print()
        return
    
    # Cleanup old downloads
    if args.cleanup is not None:
        print(f"🗑️  Cleaning up downloads older than {args.cleanup} days...")
        removed_count = downloader.cleanup_downloads(args.cleanup)
        print(f"✅ Removed {removed_count} old files")
        return
    
    # Validate URL
    if not args.url:
        print("❌ YouTube URL is required")
        print("💡 Examples:")
        print("   • Standard download: uv run python -m tools.youtube_download \"URL\" --type video --quality 1080p")
        print("   • High-quality 4K:   uv run python -m tools.youtube_download \"URL\" --type video --quality 4k --codec vp9")
        print("   • Maximum quality:   uv run python -m tools.youtube_download \"URL\" --type video --quality highest --show-formats")
        print("   • Use --recommended to see educational channels")
        return
    
    print("🌳 MoneyTree: YouTube Content Downloader")
    print("=" * 50)
    print(f"🔗 URL: {args.url}")
    print(f"📥 Type: {args.type}")
    print(f"📁 Output: {args.output_dir}")
    
    # Get video info first
    print("\\n1️⃣ Getting video information...")
    video_info = downloader.get_video_info(args.url)
    
    if not video_info:
        print("❌ Failed to get video information")
        return
    
    print(f"✅ Video found!")
    print(f"   📺 Title: {video_info.title}")
    print(f"   👤 Uploader: {video_info.uploader}")
    print(f"   ⏱️  Duration: {video_info.duration:.0f} seconds ({video_info.duration/60:.1f} minutes)")
    print(f"   👀 Views: {video_info.view_count:,}")
    print(f"   📅 Upload Date: {video_info.upload_date}")
    print(f"   🎥 Available Formats: {', '.join(video_info.available_formats[:5])}{'...' if len(video_info.available_formats) > 5 else ''}")
    
    # Test format selection
    if args.test_formats:
        print("\\n🧪 Testing format selection for different quality settings:")
        print("=" * 60)
        
        quality_tests = ['720p', '1080p', '1440p', '2160p', '4k', 'high', 'highest']
        
        for test_quality in quality_tests:
            test_config = DownloadConfig(codec_preference=args.codec, prefer_60fps=getattr(args, '60fps', False))
            test_downloader = YouTubeDownloader(test_config)
            format_string = test_downloader._get_video_format_string(test_quality)
            print(f"🎯 {test_quality:>8}: {format_string}")
        
        print("\\n💡 Use --show-formats to see what's actually available for this video")
        return

    if args.type == 'info':
        print("\\n📋 Info only mode - no download performed")
        print("💡 Use --type video or --type audio to download")
        print("💡 Use --test-formats to see format selection logic")
        return
    
    # Perform download
    print(f"\\n2️⃣ Downloading {args.type}...")
    
    if args.type == 'video':
        result = downloader.download_video(
            args.url,
            output_filename=args.output,
            quality=args.quality
        )
    else:  # audio
        result = downloader.download_audio(
            args.url,
            output_filename=args.output,
            quality=args.quality
        )
    
    if result['success']:
        print(f"✅ {args.type.title()} downloaded successfully!")
        print(f"💾 Saved to: {result['output_path']}")
        print(f"📦 File size: {result['file_size']:,} bytes ({result['file_size']/(1024*1024):.1f} MB)")
        print(f"🎯 Requested quality: {result['quality']}")
        print(f"📄 Format: {result['format']}")
        print(f"⏱️  Duration: {result['duration']:.0f} seconds")
        
        # Show actual video quality if available
        if 'actual_quality' in result and args.type == 'video':
            actual = result['actual_quality']
            print(f"\\n📊 Actual Video Quality:")
            print(f"   Resolution: {actual['resolution']}")
            print(f"   Height: {actual['height']}p")
            print(f"   FPS: {actual['fps']}")
            print(f"   Video Codec: {actual['vcodec']}")
            print(f"   Audio Codec: {actual['acodec']}")
            print(f"   Bitrate: {actual['bitrate']} kbps")
            
            # Quality assessment
            if actual['height'] != 'Unknown' and str(actual['height']).isdigit():
                height = int(actual['height'])
                if height >= 2160:
                    quality_assessment = "🌟 Excellent (4K+)"
                elif height >= 1440:
                    quality_assessment = "⭐ Very Good (1440p+)"
                elif height >= 1080:
                    quality_assessment = "👍 Good (1080p+)"
                elif height >= 720:
                    quality_assessment = "📺 Standard (720p+)"
                else:
                    quality_assessment = "📱 Low (<720p)"
                print(f"   Assessment: {quality_assessment}")
        
        # Provide Windows path
        windows_path = str(result['output_path']).replace('/mnt/c/', 'C:\\\\').replace('/', '\\\\')
        print(f"🪟 Windows path: {windows_path}")
        
        # Usage suggestions
        if args.type == 'video':
            print(f"\\n💡 Usage suggestions:")
            print(f"   • Use as template: uv run python -m demos.wikipedia_video \"Topic\" --template \"{result['output_path']}\"")
            print(f"   • Edit with VideoClip: VideoClip(\"{result['output_path']}\").add_captions(...)")
            print(f"\\n🎯 Quality tips:")
            print(f"   • For 4K: --quality 4k --codec vp9")
            print(f"   • For maximum: --quality highest --show-formats")
            print(f"   • For web streaming: --quality high --60fps")
        else:
            print(f"\\n💡 Usage suggestions:")
            print(f"   • Background music: uv run python -m demos.wikipedia_video \"Topic\" --music \"{result['output_path']}\"")
            print(f"   • Audio analysis: Use with librosa for speech/music analysis")
    else:
        print(f"❌ {args.type.title()} download failed: {result.get('error', 'Unknown error')}")
        print("💡 Try a different quality setting or check the URL")
    
    # Show download history summary
    downloads = downloader.get_download_history()
    video_count = len([d for d in downloads if d['type'] == 'video'])
    audio_count = len([d for d in downloads if d['type'] == 'audio'])
    print(f"\\n📊 Download Summary:")
    print(f"   Total files: {len(downloads)} ({video_count} videos, {audio_count} audio)")
    total_size = sum(d['size'] for d in downloads)
    print(f"   Total size: {total_size/(1024*1024):.1f} MB")
    print(f"   Videos: {downloader.video_dir}")
    print(f"   Audio: {downloader.audio_dir}")


if __name__ == "__main__":
    main()
