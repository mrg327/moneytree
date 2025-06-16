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
    parser.add_argument('--quality', help='Quality setting (720p, 1080p for video; 128, 192 for audio)')
    parser.add_argument('--output', help='Custom output filename (without extension)')
    parser.add_argument('--format', help='Output format (mp4, webm for video; mp3, m4a for audio)')
    parser.add_argument('--output-dir', default='downloads', help='Output directory')
    parser.add_argument('--list-downloads', action='store_true', help='List previously downloaded files')
    parser.add_argument('--cleanup', type=int, metavar='DAYS', 
                       help='Clean up downloads older than DAYS')
    parser.add_argument('--recommended', action='store_true', 
                       help='Show recommended educational channels')
    
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
        print("💡 Example: python demo_youtube_download.py \"https://www.youtube.com/watch?v=VIDEO_ID\" --type video")
        print("💡 Use --recommended to see educational channels")
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
    
    if args.type == 'info':
        print("\\n📋 Info only mode - no download performed")
        print("💡 Use --type video or --type audio to download")
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
        print(f"🎯 Quality: {result['quality']}")
        print(f"📄 Format: {result['format']}")
        print(f"⏱️  Duration: {result['duration']:.0f} seconds")
        
        # Provide Windows path
        windows_path = str(result['output_path']).replace('/mnt/c/', 'C:\\\\').replace('/', '\\\\')
        print(f"🪟 Windows path: {windows_path}")
        
        # Usage suggestions
        if args.type == 'video':
            print(f"\\n💡 Usage suggestions:")
            print(f"   • Use as template: python demo_video.py \"Topic\" --template \"{result['output_path']}\"")
            print(f"   • Edit with VideoClip: VideoClip(\"{result['output_path']}\").add_captions(...)")
        else:
            print(f"\\n💡 Usage suggestions:")
            print(f"   • Background music: python demo_video.py \"Topic\" --music \"{result['output_path']}\"")
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