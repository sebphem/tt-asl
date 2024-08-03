<!DOCTYPE html>
<html>
<head>
    <title>Uploaded Image</title>
    <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='styles.css') }}">
    <style>
        .container {
            text-align: center; /* Center-aligns inline and inline-block elements within the container */
        }

        #image-heading {
            margin-bottom: 10px; /* Reduce the space between the heading and the image */
        }

        .image-container {
            position: relative;
            display: inline-block; /* Center the image-container while keeping it as an inline-block element */
            margin: 20px auto; /* Center the image container horizontally and add margin */
        }

        #uploaded-image {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
        }

        #label-overlay {
            position: absolute;
            bottom: 15px; /* Adjust to move the label further up from the bottom edge */
            left: 10px; /* Keep the label 10px from the left edge */
            font-size: 2em; /* Increase size of the label */
            font-weight: bold;
            color: white;
            background-color: rgba(0, 0, 0, 0.5); /* Semi-transparent background */
            padding: 5px; /* Padding around the text */
            box-sizing: border-box; /* Include padding in the element's width and height */
            display: none; /* Hide the label by default */
        }

        .buttons {
            margin-top: 20px; /* Space above the buttons */
            text-align: center; /* Center the buttons section */
        }

        .buttons form, .buttons a {
            display: inline-block;
            margin: 0 10px; /* Space between the button and link */
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <img src="{{ url_for('static', filename='logo.png') }}" alt="Logo" class="logo">
            <h1>TT-asl</h1>
        </header>
        <h2 id="image-heading">Uploaded Image</h2> <!-- Add an ID to the heading for easy access -->
        <div class="image-container">
            <img id="uploaded-image" src="{{ url_for('serve_file', filename=filename) }}" alt="Uploaded Image">
            <div id="label-overlay"></div> <!-- Overlay container for the label -->
        </div>
        <div class="buttons">
            <form id="run-ml-form" action="{{ url_for('run_ml') }}" method="post">
                <input type="hidden" name="filename" value="{{ filename }}">
                <button type="submit" class="button blue-button">Run ML</button>
            </form>
            <a href="{{ url_for('index') }}" class="button red-button">Upload Another</a>
        </div>
    </div>
    <script>
        document.getElementById('run-ml-form').addEventListener('submit', function(event) {
            event.preventDefault(); // Prevent the default form submission
            var formData = new FormData(this);

            fetch(this.action, {
                method: this.method,
                body: formData
            }).then(response => response.json())
              .then(data => {
                  if (data.error) {
                      alert(data.error); // Handle error if present
                      return;
                  }
                  
                  // Update the image source to show the processed image
                  document.getElementById('uploaded-image').src = data.image_path;
                  document.getElementById('uploaded-image').alt = "Processed Image"; // Update the alt text

                  // Update the heading text to "Processed Image"
                  document.getElementById('image-heading').innerText = 'Processed Image';

                  // Display the label overlay on the image
                  var labelOverlay = document.getElementById('label-overlay');
                  labelOverlay.innerHTML = data.label; // Set the label from the response
                  labelOverlay.style.display = 'block'; // Show the label overlay

                  // Delete the original image
                  fetch('{{ url_for('delete_file') }}', {
                      method: 'POST',
                      headers: {
                          'Content-Type': 'application/x-www-form-urlencoded'
                      },
                      body: new URLSearchParams({ 'filename': '{{ filename }}' })
                  }).then(response => response.json())
                    .then(deleteData => {
                        if (deleteData.error) {
                            console.error('Delete Error:', deleteData.error);
                        } else {
                            console.log('File deleted successfully');
                        }
                    }).catch(error => {
                        console.error('Error:', error);
                    });
              }).catch(error => {
                  console.error('Error:', error);
                  alert('An error occurred while processing the request.');
              });
        });
    </script>
</body>
</html>
