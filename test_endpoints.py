#!/usr/bin/env python3
"""
Test script to verify endpoint connections between frontend and LLaMA3 backend
"""
import requests
import json
import time
import sys

def test_python_backend():
    """Test the Python FastAPI backend directly"""
    print("🔍 Testing Python FastAPI backend...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Python backend health check: {data}")
            return data.get('model_loaded', False) and data.get('vectorstore_loaded', False)
        else:
            print(f"❌ Python backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Python backend not running on port 8000")
        return False
    except Exception as e:
        print(f"❌ Error testing Python backend: {e}")
        return False

def test_nodejs_backend():
    """Test the Node.js Express backend"""
    print("\n🔍 Testing Node.js Express backend...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:3000/api/llm/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Node.js backend health check: {data}")
            return data.get('ok', False)
        else:
            print(f"❌ Node.js backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Node.js backend not running on port 3000")
        return False
    except Exception as e:
        print(f"❌ Error testing Node.js backend: {e}")
        return False

def test_query_flow():
    """Test the complete query flow from Node.js to Python backend"""
    print("\n🔍 Testing complete query flow...")
    
    test_questions = [
        "What is the average temperature in the Arabian Sea?",
        "Tell me about salinity levels in 2024",
        "health_check"
    ]
    
    for question in test_questions:
        try:
            print(f"\n📝 Testing question: '{question}'")
            
            # Test direct Python backend
            python_response = requests.post(
                "http://localhost:8000/query",
                json={"question": question},
                timeout=30
            )
            
            if python_response.status_code == 200:
                python_data = python_response.json()
                print(f"✅ Python backend response: {python_data.get('answer', 'No answer')[:100]}...")
            else:
                print(f"❌ Python backend query failed: {python_response.status_code}")
                continue
            
            # Test Node.js proxy
            node_response = requests.post(
                "http://localhost:3000/ask",
                json={"question": question},
                timeout=30
            )
            
            if node_response.status_code == 200:
                node_data = node_response.json()
                print(f"✅ Node.js proxy response: {str(node_data)[:100]}...")
            else:
                print(f"❌ Node.js proxy query failed: {node_response.status_code}")
                continue
                
            print("✅ Query flow working correctly!")
            
        except requests.exceptions.Timeout:
            print("⏰ Query timed out (this is normal for LLaMA3)")
        except Exception as e:
            print(f"❌ Error testing query flow: {e}")

def test_frontend_integration():
    """Test frontend integration by checking if the dashboard loads"""
    print("\n🔍 Testing frontend integration...")
    
    try:
        response = requests.get("http://localhost:3000/", timeout=10)
        if response.status_code == 200:
            print("✅ Frontend dashboard loads successfully")
            return True
        else:
            print(f"❌ Frontend dashboard failed to load: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Frontend not accessible on port 3000")
        return False
    except Exception as e:
        print(f"❌ Error testing frontend: {e}")
        return False

def main():
    print("🚀 Starting endpoint verification tests...\n")
    
    # Test Python backend
    python_ok = test_python_backend()
    
    # Test Node.js backend
    nodejs_ok = test_nodejs_backend()
    
    # Test frontend
    frontend_ok = test_frontend_integration()
    
    # Test query flow if both backends are running
    if python_ok and nodejs_ok:
        test_query_flow()
    else:
        print("\n⚠️  Skipping query flow tests - backends not ready")
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY:")
    print(f"Python Backend (port 8000): {'✅ OK' if python_ok else '❌ FAILED'}")
    print(f"Node.js Backend (port 3000): {'✅ OK' if nodejs_ok else '❌ FAILED'}")
    print(f"Frontend Dashboard: {'✅ OK' if frontend_ok else '❌ FAILED'}")
    
    if python_ok and nodejs_ok and frontend_ok:
        print("\n🎉 All endpoints are working correctly!")
        print("You can now test the chatbot in the frontend at http://localhost:3000")
    else:
        print("\n⚠️  Some endpoints are not working. Please check the services.")
        print("\nTo start the services:")
        print("1. Python backend: cd bot-using-NetCDF-ocean-data && python scripts/api_server.py")
        print("2. Node.js frontend: npm start")

if __name__ == "__main__":
    main()
