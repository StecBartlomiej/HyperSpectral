#include "Logger.hpp"
#include "Image.hpp"
#include "Gui.hpp"

#include "EntityComponentSystem.hpp"
#include "Components.hpp"


Coordinator coordinator{};


int main()
{
    Logger::Init(spdlog::level::trace);
    GLFWwindow* window = CreateWindow();

    RegisterComponents();
    auto *image_window_sys = RegisterGuiImageWindow();
    auto *pca_window_sys = RegisterPCAWindow();
    auto *threshold_window_sys = RegisterGuiThreshold();

    auto img_1 = CreateImage(
        FilesystemPaths{
            .envi_header = R"(E:\Praca inzynierska\HSI images\hyperspectralData.hdr)",
            .img_data =  R"(E:\Praca inzynierska\HSI images\hyperspectralData.dat)"}
        );

    auto img_2 = CreateImage(
        FilesystemPaths{
            .envi_header = R"(E:\Praca inzynierska\HSI images\img2.hdr)",
            .img_data =  R"(E:\Praca inzynierska\HSI images\img2.dat)"}
        );

    bool show_demo_window = true;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0)
        {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::DockSpaceOverViewport(ImGui::GetMainViewport()->ID, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);

        image_window_sys->Show();
        threshold_window_sys->Show();
        pca_window_sys->Show();


        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        GLFWwindow* backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
