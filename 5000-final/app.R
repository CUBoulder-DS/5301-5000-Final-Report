# libraries
library(shiny)
library(tidyverse)
library(shinythemes)
library(DT)

# data
# location of data
utk_1 <- "https://raw.githubusercontent.com/CUBoulder-DS/5301-5000-Final-Report/main/data/UTK/UTKpart1.csv"
utk_2 <- "https://raw.githubusercontent.com/CUBoulder-DS/5301-5000-Final-Report/main/data/UTK/UTKpart2.csv"
utk_3 <- "https://raw.githubusercontent.com/CUBoulder-DS/5301-5000-Final-Report/main/data/UTK/UTKpart3.csv"
# download data
df_1 <- read_csv(utk_1, col_select = c(src_age, src_race, src_gender))
df_2 <- read_csv(utk_2, col_select = c(src_age, src_race, src_gender))
df_3 <- read_csv(utk_3, col_select = c(src_age, src_race, src_gender))
# combine data
df <- rbind(df_1, df_2, df_3)
# clean data
mydata <- df %>%
  filter(src_age != '0' & src_race != '0' & src_gender != '0') %>%
  rename(Age = src_age, Race = src_race, Gender = src_gender) 

# attach allows use of column names without calling full table
attach(mydata)

# server input
server = function(input, output, session) {
  
  # underlying data datatable
  output$originalData <- DT::renderDataTable(DT::datatable(mydata))
  
  output$underlyingDownload = downloadHandler(
    filename = "underlying.csv",
    content = function(file) {
      write.csv(mydata, file)
    }
  )
  
  # checkbox selection for race
  output$raceSelect <- renderPrint({ input$checkGroupRace })
  
  # checkbox selection for gender
  output$genderSelect <- renderPrint({ input$checkGroupGender })
  
  # selectbox for race, gender or none
  output$subDensity <- renderPrint({ input$subDensitySelect })
  
  # slider for age
  output$ageRange <- renderPrint({ input$sliderAge })
  
  # slider for alpha
  output$alphaRange <- renderPrint({ input$sliderAlpha })
  
  # plot
  output$densityPlot = renderPlot({
    # age filter
    gdata <- mydata %>%
      filter(Age >= input$sliderAge[1] & Age <= input$sliderAge[2])
    
    # gender filter
    if (length(input$checkGroupGender) == 0) {
      gdata <- gdata
    } else if (length(input$checkGroupGender) == 1 & input$checkGroupGender[1] == 'Female') {
      gdata <- gdata %>%
        filter(Gender == 'Female')
    } else if (length(input$checkGroupGender) == 1 & input$checkGroupGender[1] == 'Male') {
      gdata <- gdata %>%
        filter(Gender == 'Male')
    }
    
    # race filter
    races <- c('Asian', 'Black', 'Indian', 'Other', 'White')
    if (length(input$checkGroupRace) == 0 | length(input$checkGroupRace) == 5) {
      gdata <- gdata
    } else {
      gdata <- gdata %>%
        filter(Race %in% races[which(races %in% input$checkGroupRace)])
    }
    
    # filter for just age, or add gender or race curves
    if (input$subDensitySelect == 'None') {
      ggplot(gdata) +
        geom_density(aes(Age))
    } else if (input$subDensitySelect == 'Gender') {
      ggplot(gdata) +
        geom_density(aes(Age, fill = Gender), alpha = input$sliderAlpha)
    } else {
      ggplot(gdata) +
        geom_density(aes(Age, fill = Race), alpha = input$sliderAlpha)
    }
  })
}

ui = navbarPage(
  # theme
  theme = shinytheme("flatly"),
  
  # title
  title = "Bias in Facial Classification",
  
  # tab - exploratory analysis
  tabPanel("Exploratory Analysis",
           fluidRow(
             # Race selection
             column(4,
                    checkboxGroupInput("checkGroupRace", label = h3("Race Selection"), 
                                       choices = list("Asian" = 'Asian',
                                                      "Black" = 'Black',
                                                      "Indian" = 'Indian',
                                                      "Other" = 'Other',
                                                      "White" = 'White'),
                                       selected = c('Asian', 'Black', 'Indian', 'Other', 'White'))),
             # Gender selection
             column(4,
                    checkboxGroupInput("checkGroupGender", label = h3("Gender Selection"), 
                                       choices = list("Female" = 'Female',
                                                      "Male" = 'Male'),
                                       selected = c('Male', 'Female'))),
             # subDensity selection
             column(4,
                    selectInput("subDensitySelect", label = h3("Sub Filter"), 
                                choices = list("Race" = 'Race',
                                               "Gender" = 'Gender',
                                               "None" = 'None'),
                                selected = 'None'))
           ),
           fluidRow(
             # Age selection
             column(6,
                    sliderInput("sliderAge", label = h3("Age Range"), min = 1, 
                                max = 135, value = c(15, 90))),
             # Alpha selection
             column(6,
                    sliderInput("sliderAlpha", label = h3("Alpha Range"), min = 0, 
                                max = 1, value = 0.2)
             )),
           
           # plot
           plotOutput("densityPlot")),
  
  # tab - underlying data
  tabPanel("Underlying Data",
           # table output 
           DT::dataTableOutput("originalData"),
           # table download
           downloadButton(outputId = "underlyingDownload", label = "Download Table"))
  
)

shinyApp(ui = ui, server = server)