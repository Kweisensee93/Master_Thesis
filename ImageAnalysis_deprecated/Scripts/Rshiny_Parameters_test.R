library(shiny)
library(reticulate)
library(yaml)

### Configuration
yaml_file <- "Outline_Parameters.yaml"
initial_config <- read_yaml(yaml_file)

python_script_01 <- "Outline_Pipeline_01.py"

### UI
ui <- fluidPage(
  titlePanel("Landmark Detection Pipeline Parameters"),
  
  #-----------------------------------------------------------------------------
  # Active buttons
  fluidRow(
    column(
      12,
      actionButton("update_yaml_btn", "Update Parameters"),
      actionButton("check_btn", "Check")
    )
  ),
  
  sidebarLayout(
    sidebarPanel(
      # Image path input
      textInput(
        "image_path",
        "Image Path:",
        value = initial_config$image$path
      ),
      textInput(
        "image_name",
        "Image name:",
        value = initial_config$image$name
      ),
      
      # TSP file path input
      textInput(
        "tsp_path",
        "TSP File Path:", 
        value = initial_config$tps$path
      ),
      
      hr(),
      
      # -----------------------
      # CLAHE controls
      # -----------------------
      sliderInput(
        "clahe_clip",
        "CLAHE clipLimit:",
        min = 0,
        max = 10,
        step = 0.1,
        value = initial_config$parameters_01$CLAHE_default$clipLimit
      ),
      
      sliderInput(
        "clahe_gridsize",
        "CLAHE tileGridSize:",
        min = 1,
        max = 20,
        step = 1,
        value = initial_config$parameters_01$CLAHE_default$tileGridSize[1]
      ),
      
      # -----------------------
      # Blur controls
      # -----------------------
      selectInput(
        "blur_type",
        "Blur type:",
        choices = initial_config$parameters_01$Blur$valid,
        selected = initial_config$parameters_01$Blur_default$type
      ),
      
      sliderInput(
        "blur_ksize",
        "Blur kernel size (odd):",
        min = 1,
        max = 21,
        step = 2,
        value = initial_config$parameters_01$Blur_default$ksize
      )
    ),
    
    mainPanel(
      # Status message
      verbatimTextOutput("status"),
      
      # 2x2 grid for images
      fluidRow(
        column(6, 
               h4("1. Raw Image"),
               plotOutput("img_raw", height = "300px")
        ),
        column(6, 
               h4("2. Landmarks"),
               plotOutput("img_landmarks", height = "300px")
        )
      ),
      fluidRow(
        column(6, 
               h4("3. CLAHE"),
               plotOutput("img_clahe", height = "300px")
        ),
        column(6, 
               h4("4. Blur"),
               plotOutput("img_blur", height = "300px")
        )
      ),
      
      # Navigation arrows in bottom right corner
      div(
        style = "position: absolute; bottom: 20px; right: 20px;",
        actionButton("prev_btn", "←", 
                     style = "font-size: 20px; margin-right: 5px;"),
        actionButton("next_btn", "→", 
                     style = "font-size: 20px;")
      )
    )
  )
)

# --------------------------------------------------
# Server
# --------------------------------------------------
server <- function(input, output, session) {
  
  rv <- reactiveValues(
    status_message = "Ready.",
    current_step = 1,
    images_loaded = FALSE,
    output_dir = NULL
  )
  
  # Display status
  output$status <- renderText({
    rv$status_message
  })
  
  # Navigation
  observeEvent(input$prev_btn, {
    rv$current_step <- max(1, rv$current_step - 1)
    rv$status_message <- paste("Step:", rv$current_step)
  })
  
  observeEvent(input$next_btn, {
    rv$current_step <- rv$current_step + 1
    rv$status_message <- paste("Step:", rv$current_step)
  })
  
  # --------------------------------------------------
  # Update YAML parameters
  # --------------------------------------------------
  observeEvent(input$update_yaml_btn, {
    
    cfg <- read_yaml(yaml_file)
    
    # Update image + TPS (if changed)
    cfg$image$path <- input$image_path
    cfg$image$name <- input$image_name
    cfg$tps$path <- input$tsp_path
    
    # Update active parameters (NOT defaults)
    cfg$parameters_01$CLAHE$clipLimit <- input$clahe_clip
    cfg$parameters_01$CLAHE$tileGridSize <- c(input$clahe_gridsize, input$clahe_gridsize)
    cfg$parameters_01$Blur$type <- input$blur_type
    cfg$parameters_01$Blur$ksize <- input$blur_ksize
    
    write_yaml(cfg, yaml_file)
    
    rv$status_message <- "YAML updated successfully."
  })
  
  # --------------------------------------------------
  # Run Python pipeline and load images
  # --------------------------------------------------
  observeEvent(input$check_btn, {
    
    rv$status_message <- "Running pipeline..."
    rv$images_loaded <- FALSE
    
    if (!file.exists(python_script_01)) {
      rv$status_message <- paste("Python script not found:", python_script_01)
      return()
    }
    
    # Run Python script
    res <- system2(
      command = "python",
      args = python_script_01,
      stdout = TRUE,
      stderr = TRUE
    )
    
    # Get output directory from YAML
    cfg <- read_yaml(yaml_file)
    rv$output_dir <- cfg$output$path
    
    # Check if images were created
    img_files <- c("01_raw.png", "02_landmarks.png", "03_clahe.png", "04_blur.png")
    img_paths <- file.path(rv$output_dir, img_files)
    
    if (all(file.exists(img_paths))) {
      rv$images_loaded <- TRUE
      rv$status_message <- "Pipeline finished successfully. Images loaded."
    } else {
      rv$status_message <- paste(
        "Pipeline finished but some images are missing.\n",
        paste(res, collapse = "\n")
      )
    }
  })
  
  # --------------------------------------------------
  # Render images
  # --------------------------------------------------
  output$img_raw <- renderPlot({
    if (rv$images_loaded && !is.null(rv$output_dir)) {
      img_path <- file.path(rv$output_dir, "01_raw.png")
      if (file.exists(img_path)) {
        img <- png::readPNG(img_path)
        par(mar = c(0, 0, 0, 0))
        plot(0:1, 0:1, type = "n", axes = FALSE, xlab = "", ylab = "")
        rasterImage(img, 0, 0, 1, 1)
      }
    }
  })
  
  output$img_landmarks <- renderPlot({
    if (rv$images_loaded && !is.null(rv$output_dir)) {
      img_path <- file.path(rv$output_dir, "02_landmarks.png")
      if (file.exists(img_path)) {
        img <- png::readPNG(img_path)
        par(mar = c(0, 0, 0, 0))
        plot(0:1, 0:1, type = "n", axes = FALSE, xlab = "", ylab = "")
        rasterImage(img, 0, 0, 1, 1)
      }
    }
  })
  
  output$img_clahe <- renderPlot({
    if (rv$images_loaded && !is.null(rv$output_dir)) {
      img_path <- file.path(rv$output_dir, "03_clahe.png")
      if (file.exists(img_path)) {
        img <- png::readPNG(img_path)
        par(mar = c(0, 0, 0, 0))
        plot(0:1, 0:1, type = "n", axes = FALSE, xlab = "", ylab = "")
        rasterImage(img, 0, 0, 1, 1)
      }
    }
  })
  
  output$img_blur <- renderPlot({
    if (rv$images_loaded && !is.null(rv$output_dir)) {
      img_path <- file.path(rv$output_dir, "04_blur.png")
      if (file.exists(img_path)) {
        img <- png::readPNG(img_path)
        par(mar = c(0, 0, 0, 0))
        plot(0:1, 0:1, type = "n", axes = FALSE, xlab = "", ylab = "")
        rasterImage(img, 0, 0, 1, 1)
      }
    }
  })
}

# Run the app
shinyApp(ui = ui, server = server)