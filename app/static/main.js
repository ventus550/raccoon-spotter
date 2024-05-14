var fileDrag = document.getElementById("file-drag");
var fileSelect = document.getElementById("file-upload");

// Add event listeners
fileDrag.addEventListener("dragover", fileDragHover, false);
fileDrag.addEventListener("dragleave", fileDragHover, false);
fileDrag.addEventListener("drop", fileSelectHandler, false);
fileSelect.addEventListener("change", fileSelectHandler, false);

function fileDragHover(e) {
  // Prevent default behaviour
  e.preventDefault();
  e.stopPropagation();

  fileDrag.className = e.type === "dragover" ? "upload-box dragover" : "upload-box";
}

function fileSelectHandler(e) {
  // Handle file selecting
  console.log("File selected:", e.target.files);
  var files = e.target.files || e.dataTransfer.files;
  fileDragHover(e);
  for (var i = 0, f; f = files[i]; i++) {
    previewFile(f);
  }
}

//========================================================================
// Web page elements for functions to use
//========================================================================

var imagePreview = document.getElementById("image-preview");
var imageDisplay = document.getElementById("image-display");
var uploadCaption = document.getElementById("upload-caption");
var predResult = document.getElementById("pred-result");
var loader = document.getElementById("loader");

//========================================================================
// Main button events
//========================================================================

function submitImage() {
  // Action for the submit button
  console.log("Submit");

  if (!imageDisplay.src || !imageDisplay.src.startsWith("data")) {
    window.alert("Please select an image before submitting.");
    return;
  }

  loader.classList.remove("hidden");
  imageDisplay.classList.add("loading");

  // Call the predict function of the backend
  predictImage({ image: imageDisplay.src });
}

function clearImage() {
  console.log("Clearing image and resetting all related fields and UI elements.");

  // Additional code for debugging
  fileSelect.value = "";

  console.log("File select value cleared:", fileSelect.value);

  // Reset UI elements
  imagePreview.src = "";
  imageDisplay.src = "";
  predResult.innerHTML = "";
  hide(imagePreview);
  hide(imageDisplay);
  hide(loader);
  hide(predResult);
  show(uploadCaption);

  imageDisplay.classList.remove("loading");

  console.log("UI elements reset and hidden as necessary.");
}

function fileSelectHandler(e) {
  console.log("Handling file select or drop event.");
  var files = e.target.files || e.dataTransfer.files;

  console.log("Files received:", files.length);
  fileDragHover(e);
  
  if (!files.length) {
    console.log("No files detected, check the file input or drop event.");
    return;
  }

  for (var i = 0, f; f = files[i]; i++) {
    console.log("Previewing file:", f.name);
    previewFile(f);
  }
}



function previewFile(file) {
  // Show the preview of the image
  console.log("Loading image:", file.name);
  
  var reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = () => {
    imagePreview.src = reader.result;

    show(imagePreview);
    hide(uploadCaption);

    // Reset
    predResult.innerHTML = "";
    imageDisplay.classList.remove("loading");
    imageDisplay.src = reader.result; // Display the image in `imageDisplay` element
  };
}

function predictImage(imageData) {
  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(imageData)
  })
    .then(response => response.json())
    .then(data => {
      displayResult(data);
    })
    .catch(err => {
      console.error("An error occurred:", err.message);
      window.alert("Oops! Something went wrong.");
    })
    .finally(() => {
      hide(loader);
      imageDisplay.classList.remove("loading");
    });
}

function displayResult(data) {
  console.log("Received data:", data);

  if (data.image_base64) {
    console.log("Setting image source.");
    imageDisplay.src = `data:image/png;base64,${data.image_base64}`;
    show(imageDisplay); 
  } else {
    console.log("No image base64 data found.");
    hide(imageDisplay);
  }

  show(predResult);
}


function show(el) {
  console.log("Showing element:", el.id);
  el.classList.remove("hidden");
}

function hide(el) {
  console.log("Hiding element:", el.id);
  el.classList.add("hidden");
}

