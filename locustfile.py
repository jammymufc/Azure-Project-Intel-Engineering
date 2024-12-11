from locust import HttpUser, task, between
import json

class InferUser(HttpUser):
    # Adjust wait time between tasks (1-5 seconds)
    wait_time = between(1, 5)

    @task
    def make_inference(self):
        # Example input data for the model 
        input_data = {
            "input_data": [
                [
                0,
                0,
                0
                ]
            ],
            "params": {}
            }

        # Set the headers for authorization and content type
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer 5HtE53VSSfFgFS4XHodNepiMj5Z0IykU'
        }

        # Make the POST request to the model's /score endpoint
        response = self.client.post(
            "https://projectworkspace-igvxr.ukwest.inference.ml.azure.com/score", 
            json=input_data,
            headers=headers
        )

        # Print response for debugging 
        if response.status_code != 200:
            print(f"Error response: {response.text}")
        else:
            print(f"Success response: {response.json()}")

        # Assert that the response status code is 200 (success)
        assert response.status_code == 200, f"Failed with status code {response.status_code}"
