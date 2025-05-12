# test_hf_api.py
import os
import traceback
from dotenv import load_dotenv
from huggingface_hub import HfApi, InferenceClient
from huggingface_hub.utils import HfHubHTTPError
# Attempt to import the internal provider mapping for inspection
try:
    from huggingface_hub.inference._providers import DEFAULT_PROVIDER_MAPPING
except ImportError:
    DEFAULT_PROVIDER_MAPPING = "Could not import DEFAULT_PROVIDER_MAPPING"
try:
    from huggingface_hub.inference._providers.hf_inference_endpoints import HfInferenceEndpointsServerlessHelper
except ImportError:
    HfInferenceEndpointsServerlessHelper = "Could not import HfInferenceEndpointsServerlessHelper"


load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    print("HUGGINGFACEHUB_API_TOKEN not found in .env file.")
else:
    print(f"Using HF Token: {HF_TOKEN[:5]}...{HF_TOKEN[-5:]}")

    print("\n--- Debugging Hugging Face Provider Mappings ---")
    print(f"DEFAULT_PROVIDER_MAPPING from library: {DEFAULT_PROVIDER_MAPPING}")
    print(f"HfInferenceEndpointsServerlessHelper type: {type(HfInferenceEndpointsServerlessHelper)}")


    print("\n--- Testing with InferenceClient ---")
    model_id = "gpt2"
    #model_id = "distilgpt2" # Try this
# Or even a more specific task if you know one, but distilgpt2 is fine

    try:
        print(f"Initializing InferenceClient for model: {model_id}...")
        # Initialize client without specifying a provider to see what it defaults to
        client = InferenceClient(token=HF_TOKEN, timeout=30)

        # Let's try to inspect the client's internal provider mapping if possible
        # This is highly internal and might change between versions
        try:
            print(f"Client's internal provider after init: {client.provider}")
            # The actual mapping might be deeper or constructed differently
            # This part is speculative for debugging:
            if hasattr(client, "_provider_helpers"):
                 print(f"Client's _provider_helpers: {client._provider_helpers}")
            if hasattr(client, "helpers_history"): # Older versions might have used this
                 print(f"Client's helpers_history: {client.helpers_history}")

        except Exception as inspect_err:
            print(f"Could not inspect client's internal provider details: {inspect_err}")


        print(f"Attempting to query model: {model_id} with a 30s timeout...")
        response = client.text_generation(
            prompt="What is the capital of France?",
            model=model_id, # Explicitly pass model here again
            max_new_tokens=50,
        )
        print(f"Response from {model_id}:")
        print(response)
    except HfHubHTTPError as http_err:
        print(f"HfHubHTTPError with InferenceClient for {model_id}: {http_err}")
        print(f"Response content from HF if any: {http_err.response.content if http_err.response else 'No response object'}")
    except ConnectionError as conn_err:
        print(f"ConnectionError with InferenceClient for {model_id}: {conn_err}")
    except TimeoutError as timeout_err:
        print(f"TimeoutError with InferenceClient for {model_id}: {timeout_err}")
    except StopIteration as si_err: # Catching StopIteration explicitly
        print(f"StopIteration caught: {si_err}")
        print("This likely means the provider_mapping inside InferenceClient is empty or misconfigured.")
        print("Full traceback for StopIteration:")
        traceback.print_exc()
    except Exception as e:
        print(f"Generic error with InferenceClient for {model_id}: {e}")
        print("Full traceback:")
        traceback.print_exc()