"""
Example Python script demonstrating how a FinTech platform integrates TrustID.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client_sdk.trustid import TrustIDClient


def main():
    print("=== TrustID Python Client SDK Demo ===")
    client = TrustIDClient(base_url="http://127.0.0.1:8000")

    # 1. Health check
    try:
        health = client.check_health()
        print(f"[*] Engine Status: {health['status']}")
    except Exception as e:
        print(f"[!] Server offline: {e}")
        return

    # 2. Initialize Session
    session = client.create_session(user_id="customer_bank_8921", document_type="AADHAAR")
    session_id = session["session_id"]
    print(f"[*] KYC Session created: {session_id}")
    print(f"[*] Verification Code: {session['verification_code']}")
    for step in session["challenges"]:
        print(f"    - Step {step['step_id']}: {step['instruction']}")

    print("\nSDK ready for streaming frames and document upload.")


if __name__ == "__main__":
    main()
