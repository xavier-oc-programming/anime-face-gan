from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'ok'
    assert 'model_loaded' in data
    assert 'latent_dim' in data


def test_generate_default():
    response = client.post('/generate', json={'n': 16})
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert 'image_base64' in data
        assert data['n_generated'] == 16
        assert len(data['image_base64']) > 100


def test_generate_single():
    response = client.get('/generate/single')
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        assert response.headers['content-type'] == 'image/png'


def test_generate_too_many():
    response = client.post('/generate', json={'n': 100})
    assert response.status_code == 422


def test_samples_endpoint():
    response = client.get('/api/samples')
    assert response.status_code == 200
    assert isinstance(response.json(), list)
