# def create_figure(self, plot_function, data, n_subplot=1):
    #
    #     n_rows = n_subplot
    #     n_cols = self.a.n_monkey if self.target_monkey is None else 1
    #     fig, axes = plt.subplots(nrows=n_rows,
    #                              ncols=n_cols,
    #                              figsize=(6*n_cols, 6*n_rows))
    #
    #     if self.target_monkey is None and self.a.n_monkey > 1:
    #         if len(axes.shape) > 1:
    #             axes = axes.T
    #         for i, m in enumerate(self.a.monkeys):
    #             plot_function(axes[i], data[m])
    #
    #     else:
    #         plot_function(axes, data[self.target_monkey])
    #
    #     plt.tight_layout()
    #     self.pdf.savefig(fig)
    #     plt.close(fig)
    #
    # def create_pdf(self, monkey=None):
    #
    #     self.target_monkey = monkey
    #
    #     # Define the path
    #     pdf_path = os.path.join(
    #         FIG_FOLDER,
    #         f"results{monkey if monkey is not None else ''}"
    #         f"{self.fig_suffix}")
    #
    #     print(f"Creating the figure '{pdf_path}'...")
    #
    #     # Create the pdf
    #     self.pdf = PdfPages(pdf_path)
    #
    #     # Fig: Info
    #     self.create_figure(
    #         plot_function=plot.info.fig_info,
    #         data=self.a.info_data)
    #
    #     # Fig: Control
    #     self.create_figure(
    #         plot_function=plot.control.plot,
    #         data=self.a.control_data)
    #
    #     # Fig: Control sigmoid
    #     self.create_figure(
    #         plot_function=plot.control_sigmoid.plot,
    #         data=self.a.control_sigmoid_data,
    #         n_subplot=len(CONTROL_CONDITIONS))
    #
    #     # Fig: Freq risky choice against expected value
    #     self.create_figure(
    #         plot_function=plot.freq_risk.plot,
    #         data=self.a.freq_risk_data)
    #
    #     # Fig: Utility function
    #     self.create_figure(
    #         plot_function=plot.utility.plot,
    #         data=self.a.cpt_fit)
    #
    #     # Fig: Probability distortion
    #     self.create_figure(
    #         plot_function=plot.probability_distortion.plot,
    #         data=self.a.cpt_fit)
    #
    #     # Fig: Precision
    #     self.create_figure(
    #         plot_function=plot.precision.plot,
    #         data=self.a.cpt_fit)
    #
    #     # Fig: Best param history
    #     self.create_figure(
    #         plot_function=plot.history_best_param.plot,
    #         data=self.a.hist_best_param_data,
    #         n_subplot=len(self.a.class_model.param_labels))
    #
    #     # Fig: Control history
    #     self.create_figure(
    #         plot_function=plot.history_control.plot,
    #         data=self.a.hist_control_data,
    #         n_subplot=len(CONTROL_CONDITIONS))
    #
    #     self.pdf.close()
    #     self.target_monkey = None
    #
    #     print(f"Figure '{pdf_path}' created.\n")

    # def create_best_param_distrib_and_lls_distrib(self):
    #
    #     # Define the path
    #     fig_path = os.path.join(
    #         FIG_FOLDER,
    #         f"best_param_distrib{self.fig_suffix}")
    #
    #     print(f"Creating the figure '{fig_path}'...")
    #
    #     plot.best_param_distrib.plot(
    #         self.a.cpt_fit, fig_path=fig_path,
    #         param_labels=self.a.class_model.param_labels)
    #
    #     fig_path_lls = os.path.join(
    #         FIG_FOLDER,
    #         f"LLS_distrib{self.fig_suffix}")
    #
    #     fig_path_bic = os.path.join(
    #         FIG_FOLDER,
    #         f"BIC_distrib{self.fig_suffix}.pdf")
    #
    #     print(f"Creating the figure '{fig_path_lls}'...")
    #     print(f"Creating the figure '{fig_path_bic}'...")
    #     plot.LLS_BIC_distrib.plot(self.a.cpt_fit, fig_path_lls=fig_path_lls,
    #                               fig_path_bic=fig_path_bic)