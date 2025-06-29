"""Tests for YouTube upload functionality."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from googleapiclient.errors import HttpError

from lib.upload.youtube import YouTubeClient


class TestYouTubeClient:
    """Test cases for YouTubeClient."""
    
    def test_init_with_api_key(self):
        """Test YouTubeClient initialization with API key."""
        client = YouTubeClient(api_key="test_key")
        assert client.api_key == "test_key"
    
    def test_init_with_env_var(self):
        """Test YouTubeClient initialization with environment variable."""
        with patch.dict(os.environ, {'YOUTUBE_API_KEY': 'env_key'}):
            client = YouTubeClient()
            assert client.api_key == "env_key"
    
    def test_init_no_api_key(self):
        """Test YouTubeClient initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="YouTube API key must be provided"):
                YouTubeClient()
    
    @patch('lib.upload.youtube.build')
    def test_authenticate_api_key_only(self, mock_build):
        """Test authentication with API key only."""
        mock_youtube = Mock()
        mock_build.return_value = mock_youtube
        
        client = YouTubeClient(api_key="test_key")
        result = client.authenticate()
        
        assert result is True
        assert client.youtube == mock_youtube
        mock_build.assert_called_once_with('youtube', 'v3', developerKey='test_key')
    
    @patch('lib.upload.youtube.build')
    @patch('lib.upload.youtube.Flow')
    @patch('lib.upload.youtube.Credentials')
    @patch('os.path.exists')
    @patch('builtins.open')
    def test_authenticate_oauth2_new_credentials(self, mock_open, mock_exists, 
                                               mock_credentials, mock_flow, mock_build):
        """Test OAuth2 authentication with new credentials."""
        # Setup mocks
        mock_exists.return_value = False
        mock_flow_instance = Mock()
        mock_flow.from_client_secrets_file.return_value = mock_flow_instance
        mock_flow_instance.authorization_url.return_value = ("http://auth.url", None)
        mock_credentials_instance = Mock()
        mock_flow_instance.credentials = mock_credentials_instance
        mock_youtube = Mock()
        mock_build.return_value = mock_youtube
        
        client = YouTubeClient(api_key="test_key")
        
        # Mock user input
        with patch('builtins.input', return_value='auth_code'):
            result = client.authenticate(client_secrets_file="secrets.json")
        
        assert result is True
        assert client.youtube == mock_youtube
        assert client.credentials == mock_credentials_instance
        mock_flow_instance.fetch_token.assert_called_once_with(code='auth_code')
    
    @patch('lib.upload.youtube.build')
    def test_authenticate_failure(self, mock_build):
        """Test authentication failure."""
        mock_build.side_effect = Exception("Auth failed")
        
        client = YouTubeClient(api_key="test_key")
        result = client.authenticate()
        
        assert result is False
        assert client.youtube is None
    
    def test_upload_video_not_authenticated(self):
        """Test upload_video without authentication."""
        client = YouTubeClient(api_key="test_key")
        result = client.upload_video("video.mp4", "Test Title")
        
        assert result is None
    
    def test_upload_video_file_not_found(self):
        """Test upload_video with non-existent file."""
        client = YouTubeClient(api_key="test_key")
        client.youtube = Mock()
        client.credentials = Mock()
        
        with patch('os.path.exists', return_value=False):
            result = client.upload_video("nonexistent.mp4", "Test Title")
        
        assert result is None
    
    @patch('lib.upload.youtube.MediaFileUpload')
    @patch('os.path.exists')
    def test_upload_video_success(self, mock_exists, mock_media_upload):
        """Test successful video upload."""
        mock_exists.return_value = True
        mock_media = Mock()
        mock_media_upload.return_value = mock_media
        
        mock_youtube = Mock()
        mock_request = Mock()
        mock_response = {'id': 'video123', 'snippet': {'title': 'Test Title'}}
        mock_request.execute.return_value = mock_response
        mock_youtube.videos().insert.return_value = mock_request
        
        client = YouTubeClient(api_key="test_key")
        client.youtube = mock_youtube
        client.credentials = Mock()
        
        result = client.upload_video(
            video_path="test.mp4",
            title="Test Title",
            description="Test Description",
            tags=["test", "video"],
            privacy_status="private"
        )
        
        assert result == mock_response
        assert result['id'] == 'video123'
    
    @patch('lib.upload.youtube.MediaFileUpload')
    @patch('os.path.exists')
    def test_upload_video_http_error(self, mock_exists, mock_media_upload):
        """Test video upload with HTTP error."""
        mock_exists.return_value = True
        mock_media = Mock()
        mock_media_upload.return_value = mock_media
        
        mock_youtube = Mock()
        mock_request = Mock()
        mock_request.execute.side_effect = HttpError(Mock(status=400), b'Bad Request')
        mock_youtube.videos().insert.return_value = mock_request
        
        client = YouTubeClient(api_key="test_key")
        client.youtube = mock_youtube
        client.credentials = Mock()
        
        result = client.upload_video("test.mp4", "Test Title")
        
        assert result is None
    
    def test_update_video_not_authenticated(self):
        """Test update_video without authentication."""
        client = YouTubeClient(api_key="test_key")
        result = client.update_video("video123", title="New Title")
        
        assert result is False
    
    def test_update_video_success(self):
        """Test successful video update."""
        mock_youtube = Mock()
        
        # Mock the list response
        mock_list_response = {
            'items': [{
                'snippet': {
                    'title': 'Old Title',
                    'description': 'Old Description',
                    'tags': ['old', 'tags']
                }
            }]
        }
        mock_youtube.videos().list().execute.return_value = mock_list_response
        
        # Mock the update response
        mock_update_response = {'id': 'video123'}
        mock_youtube.videos().update().execute.return_value = mock_update_response
        
        client = YouTubeClient(api_key="test_key")
        client.youtube = mock_youtube
        client.credentials = Mock()
        
        result = client.update_video("video123", title="New Title", tags=["new", "tags"])
        
        assert result is True
    
    def test_update_video_not_found(self):
        """Test update_video with non-existent video."""
        mock_youtube = Mock()
        mock_youtube.videos().list().execute.return_value = {'items': []}
        
        client = YouTubeClient(api_key="test_key")
        client.youtube = mock_youtube
        client.credentials = Mock()
        
        result = client.update_video("nonexistent", title="New Title")
        
        assert result is False
    
    def test_get_upload_status_not_authenticated(self):
        """Test get_upload_status without authentication."""
        client = YouTubeClient(api_key="test_key")
        result = client.get_upload_status("video123")
        
        assert result is None
    
    def test_get_upload_status_success(self):
        """Test successful get_upload_status."""
        mock_youtube = Mock()
        mock_response = {
            'items': [{
                'status': {'uploadStatus': 'processed'},
                'processingDetails': {'processingStatus': 'succeeded'}
            }]
        }
        mock_youtube.videos().list().execute.return_value = mock_response
        
        client = YouTubeClient(api_key="test_key")
        client.youtube = mock_youtube
        
        result = client.get_upload_status("video123")
        
        assert result == mock_response['items'][0]
    
    def test_get_channel_videos_not_authenticated(self):
        """Test get_channel_videos without authentication."""
        client = YouTubeClient(api_key="test_key")
        result = client.get_channel_videos()
        
        assert result == []
    
    def test_get_channel_videos_with_channel_id(self):
        """Test get_channel_videos with specific channel ID."""
        mock_youtube = Mock()
        
        # Mock channel response
        mock_channel_response = {
            'items': [{
                'contentDetails': {
                    'relatedPlaylists': {'uploads': 'playlist123'}
                }
            }]
        }
        mock_youtube.channels().list().execute.return_value = mock_channel_response
        
        # Mock playlist response
        mock_playlist_response = {
            'items': [{
                'snippet': {
                    'resourceId': {'videoId': 'video123'}
                }
            }],
            'nextPageToken': None
        }
        mock_youtube.playlistItems().list().execute.return_value = mock_playlist_response
        
        # Mock videos response
        mock_videos_response = {
            'items': [{
                'id': 'video123',
                'snippet': {
                    'title': 'Test Video',
                    'description': 'Test Description',
                    'publishedAt': '2023-01-01T00:00:00Z',
                    'tags': ['test'],
                    'thumbnails': {'high': {'url': 'http://thumb.url'}}
                },
                'statistics': {
                    'viewCount': '100',
                    'likeCount': '10',
                    'commentCount': '5'
                },
                'contentDetails': {'duration': 'PT1M30S'},
                'status': {'privacyStatus': 'public'}
            }]
        }
        mock_youtube.videos().list().execute.return_value = mock_videos_response
        
        client = YouTubeClient(api_key="test_key")
        client.youtube = mock_youtube
        
        result = client.get_channel_videos(channel_id="channel123", limit=10)
        
        assert len(result) == 1
        assert result[0]['id'] == 'video123'
        assert result[0]['title'] == 'Test Video'
        assert result[0]['view_count'] == 100
        assert result[0]['like_count'] == 10
    
    def test_get_channel_videos_own_channel(self):
        """Test get_channel_videos for authenticated user's own channel."""
        mock_youtube = Mock()
        
        # Mock own channel response
        mock_own_channel_response = {
            'items': [{'id': 'my_channel_id'}]
        }
        mock_youtube.channels().list().execute.return_value = mock_own_channel_response
        
        # Mock subsequent calls (simplified)
        mock_uploads_response = {
            'items': [{'contentDetails': {'relatedPlaylists': {'uploads': 'my_uploads'}}}]
        }
        mock_playlist_response = {'items': [], 'nextPageToken': None}
        
        # Configure mock to return different responses for different calls
        mock_youtube.channels().list.side_effect = [
            Mock(execute=Mock(return_value=mock_own_channel_response)),
            Mock(execute=Mock(return_value=mock_uploads_response))
        ]
        mock_youtube.playlistItems().list().execute.return_value = mock_playlist_response
        
        client = YouTubeClient(api_key="test_key")
        client.youtube = mock_youtube
        client.credentials = Mock()
        
        result = client.get_channel_videos()
        
        assert result == []  # No videos in playlist
    
    def test_get_channel_videos_http_error(self):
        """Test get_channel_videos with HTTP error."""
        mock_youtube = Mock()
        mock_youtube.channels().list().execute.side_effect = HttpError(
            Mock(status=404), b'Channel not found'
        )
        
        client = YouTubeClient(api_key="test_key")
        client.youtube = mock_youtube
        
        result = client.get_channel_videos(channel_id="nonexistent")
        
        assert result == []