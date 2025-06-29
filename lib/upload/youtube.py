"""YouTube API integration for video uploads and channel management."""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

logger = logging.getLogger(__name__)


class YouTubeClient:
    """Client for interacting with YouTube Data API v3."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize YouTube client.
        
        Args:
            api_key: YouTube API key. If not provided, will use YOUTUBE_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('YOUTUBE_API_KEY')
        if not self.api_key:
            raise ValueError("YouTube API key must be provided or set in YOUTUBE_API_KEY environment variable")
        
        self.youtube = None
        self.credentials = None
        
    def authenticate(self, client_secrets_file: Optional[str] = None, token_file: Optional[str] = None) -> bool:
        """Authenticate with YouTube API using OAuth2.
        
        Args:
            client_secrets_file: Path to OAuth2 client secrets JSON file
            token_file: Path to store/load OAuth2 tokens
            
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            # For upload operations, we need OAuth2 authentication
            if client_secrets_file:
                scopes = ['https://www.googleapis.com/auth/youtube.upload',
                         'https://www.googleapis.com/auth/youtube.readonly']
                
                token_file = token_file or 'youtube_token.json'
                
                # Load existing credentials if available
                if os.path.exists(token_file):
                    self.credentials = Credentials.from_authorized_user_file(token_file, scopes)
                
                # If credentials don't exist or are invalid, run OAuth flow
                if not self.credentials or not self.credentials.valid:
                    if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                        self.credentials.refresh(Request())
                    else:
                        flow = Flow.from_client_secrets_file(client_secrets_file, scopes)
                        flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
                        
                        auth_url, _ = flow.authorization_url(prompt='consent')
                        logger.info(f"Please visit this URL to authorize the application: {auth_url}")
                        
                        auth_code = input("Enter the authorization code: ")
                        flow.fetch_token(code=auth_code)
                        self.credentials = flow.credentials
                    
                    # Save credentials for future use
                    with open(token_file, 'w') as token:
                        token.write(self.credentials.to_json())
                
                self.youtube = build('youtube', 'v3', credentials=self.credentials)
            else:
                # For read-only operations, API key is sufficient
                self.youtube = build('youtube', 'v3', developerKey=self.api_key)
            
            logger.info("YouTube API authentication successful")
            return True
            
        except Exception as e:
            logger.error(f"YouTube authentication failed: {e}")
            return False
    
    def upload_video(self, 
                    video_path: str,
                    title: str,
                    description: str = "",
                    tags: Optional[List[str]] = None,
                    privacy_status: str = "private",
                    category_id: str = "22") -> Optional[Dict[str, Any]]:
        """Upload a video to YouTube.
        
        Args:
            video_path: Path to video file
            title: Video title
            description: Video description
            tags: List of tags for the video
            privacy_status: Privacy setting (private, public, unlisted)
            category_id: YouTube category ID (22 = People & Blogs)
            
        Returns:
            Dictionary with upload response or None if failed
        """
        if not self.youtube or not self.credentials:
            logger.error("Must authenticate with OAuth2 before uploading videos")
            return None
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
        
        try:
            tags = tags or []
            
            body = {
                'snippet': {
                    'title': title,
                    'description': description,
                    'tags': tags,
                    'categoryId': category_id
                },
                'status': {
                    'privacyStatus': privacy_status
                }
            }
            
            media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
            
            request = self.youtube.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            response = request.execute()
            logger.info(f"Video uploaded successfully. Video ID: {response['id']}")
            
            return response
            
        except HttpError as e:
            logger.error(f"HTTP error during video upload: {e}")
            return None
        except Exception as e:
            logger.error(f"Error uploading video: {e}")
            return None
    
    def update_video(self, video_id: str, title: Optional[str] = None, 
                    description: Optional[str] = None, tags: Optional[List[str]] = None) -> bool:
        """Update video metadata.
        
        Args:
            video_id: YouTube video ID
            title: New title (optional)
            description: New description (optional)
            tags: New tags (optional)
            
        Returns:
            True if update successful, False otherwise
        """
        if not self.youtube or not self.credentials:
            logger.error("Must authenticate with OAuth2 before updating videos")
            return False
        
        try:
            # First get current video data
            response = self.youtube.videos().list(
                part='snippet',
                id=video_id
            ).execute()
            
            if not response['items']:
                logger.error(f"Video not found: {video_id}")
                return False
            
            snippet = response['items'][0]['snippet']
            
            # Update only provided fields
            if title is not None:
                snippet['title'] = title
            if description is not None:
                snippet['description'] = description
            if tags is not None:
                snippet['tags'] = tags
            
            update_response = self.youtube.videos().update(
                part='snippet',
                body={
                    'id': video_id,
                    'snippet': snippet
                }
            ).execute()
            
            logger.info(f"Video updated successfully: {video_id}")
            return True
            
        except HttpError as e:
            logger.error(f"HTTP error updating video: {e}")
            return False
        except Exception as e:
            logger.error(f"Error updating video: {e}")
            return False
    
    def get_upload_status(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get upload and processing status of a video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary with status information or None if failed
        """
        if not self.youtube:
            logger.error("Must authenticate before checking upload status")
            return None
        
        try:
            response = self.youtube.videos().list(
                part='status,processingDetails',
                id=video_id
            ).execute()
            
            if not response['items']:
                logger.error(f"Video not found: {video_id}")
                return None
            
            return response['items'][0]
            
        except HttpError as e:
            logger.error(f"HTTP error getting upload status: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting upload status: {e}")
            return None
    
    def get_channel_videos(self, channel_id: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch videos from a YouTube channel.
        
        Args:
            channel_id: YouTube channel ID. If None, uses authenticated user's channel
            limit: Maximum number of videos to return (most recent first)
            
        Returns:
            List of dictionaries containing video metadata
        """
        if not self.youtube:
            logger.error("Must authenticate before fetching channel videos")
            return []
        
        try:
            videos = []
            
            # If no channel_id provided, get authenticated user's channel
            if not channel_id:
                if not self.credentials:
                    logger.error("Channel ID required when not using OAuth2 authentication")
                    return []
                
                channels_response = self.youtube.channels().list(
                    part='id',
                    mine=True
                ).execute()
                
                if not channels_response['items']:
                    logger.error("No channel found for authenticated user")
                    return []
                
                channel_id = channels_response['items'][0]['id']
            
            # Get channel's uploads playlist
            channel_response = self.youtube.channels().list(
                part='contentDetails',
                id=channel_id
            ).execute()
            
            if not channel_response['items']:
                logger.error(f"Channel not found: {channel_id}")
                return []
            
            uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            
            # Fetch videos from uploads playlist
            next_page_token = None
            max_results = min(50, limit) if limit else 50
            
            while True:
                playlist_response = self.youtube.playlistItems().list(
                    part='snippet',
                    playlistId=uploads_playlist_id,
                    maxResults=max_results,
                    pageToken=next_page_token
                ).execute()
                
                # Get detailed video information
                video_ids = [item['snippet']['resourceId']['videoId'] for item in playlist_response['items']]
                
                if video_ids:
                    videos_response = self.youtube.videos().list(
                        part='snippet,statistics,status,contentDetails',
                        id=','.join(video_ids)
                    ).execute()
                    
                    for video in videos_response['items']:
                        video_data = {
                            'id': video['id'],
                            'title': video['snippet']['title'],
                            'description': video['snippet']['description'],
                            'published_at': video['snippet']['publishedAt'],
                            'tags': video['snippet'].get('tags', []),
                            'view_count': int(video['statistics'].get('viewCount', 0)),
                            'like_count': int(video['statistics'].get('likeCount', 0)),
                            'comment_count': int(video['statistics'].get('commentCount', 0)),
                            'duration': video['contentDetails']['duration'],
                            'privacy_status': video['status']['privacyStatus'],
                            'thumbnail_url': video['snippet']['thumbnails'].get('high', {}).get('url', '')
                        }
                        videos.append(video_data)
                
                # Check if we've reached the limit or no more pages
                if limit and len(videos) >= limit:
                    videos = videos[:limit]
                    break
                
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token:
                    break
                
                # Adjust max_results for next request if limit is set
                if limit:
                    remaining = limit - len(videos)
                    max_results = min(50, remaining)
            
            logger.info(f"Retrieved {len(videos)} videos from channel {channel_id}")
            return videos
            
        except HttpError as e:
            logger.error(f"HTTP error fetching channel videos: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching channel videos: {e}")
            return []